# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection lifecycle: test, persist credentials, and link catalog databases."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.catalog.table_filter import FILTER_KEYS, from_connection
from auto_ontology.connectors.registry import (
    create_connector,
    invalidate_connectors_cache,
)
from auto_ontology.connectors.vault import (
    delete_secrets,
    is_vault_configured,
    write_secret,
)
from auto_ontology.server.ingestion.proxy import trigger_ingest, trigger_reset
from auto_ontology.server.chat.worker import refresh_chat_workers
from auto_ontology.dal.connections import (
    clear_connection,
    insert_connection,
    list_connections as list_connections,
    list_source_metadata,
    load_source_connection,
    normalize_source_labels,
    get_connection_storage,
    update_connection_label,
    LabelConflictError,
)

logger = logging.getLogger(__name__)

# Credentials, stripped from every payload the UI receives. An edit form
# therefore opens with these blank, which is why the update path reads blank as
# "unchanged" rather than "clear it".
SECRET_FIELDS = frozenset(
    {
        "password",
        "password_env",
        "private_key",
        "private_key_passphrase",
        "truststore_password",
        "access_token_override",
    }
)


class AmbiguousConnectionError(ValueError):
    """A legacy physical-name selector identifies more than one source."""


class SourceUnavailableError(ValueError):
    """The selected source's credential storage is unavailable."""


class ConnectionNotFoundError(LookupError, ValueError):
    """Missing managed source; retains both legacy missing-source contracts."""


def _require_managed_mode() -> None:
    if os.environ.get("CONNECTION_STRINGS", "").strip():
        raise ValueError("source_conflict: connections are environment-managed")


def _selected_source(
    connection_id: str | None, database_name: str | None
) -> dict[str, Any]:
    _require_managed_mode()
    if (connection_id is None) == (database_name is None):
        raise ValueError("invalid_source_selector")
    try:
        metadata = list_source_metadata()
    except Exception:
        raise SourceUnavailableError(
            "source_unavailable: metadata unavailable"
        ) from None
    rows = [
        row
        for row in metadata
        if row["configuration_origin"] == "managed"
        and (
            row["id"] == connection_id
            if connection_id is not None
            else row["name"] == database_name
        )
    ]
    if len(rows) > 1:
        raise AmbiguousConnectionError("ambiguous_source: use connection_id")
    if not rows:
        raise ConnectionNotFoundError("source_not_found: No connection found")
    return rows[0]


def _stored_connection(connection_id: str) -> dict[str, Any]:
    try:
        return load_source_connection(connection_id)
    except Exception:
        raise SourceUnavailableError(
            "source_unavailable: credentials unavailable"
        ) from None


def _check_label(display_label: str, connection_id: str | None = None) -> str:
    if not isinstance(display_label, str) or not display_label.strip(" "):
        raise ValueError("invalid_label: label must not be blank")
    label = display_label.strip(" ")
    try:
        normalized = normalize_source_labels([label])[0]
        metadata = list_source_metadata()
    except Exception:
        raise SourceUnavailableError(
            "source_unavailable: metadata unavailable"
        ) from None
    if any(
        row["id"] != connection_id and row["normalized_label"] == normalized
        for row in metadata
    ):
        raise LabelConflictError("label_conflict: label already exists")
    return label


def _driver_settings(connection: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "id",
        "name",
        "connection_id",
        "database_name",
        "display_label",
        "configuration_origin",
        "origin",
        "credential_ref",
        "dialect",
        "normalized_label",
        "metadata_connection_id",
        "metadata_label",
        "schema",
    }
    return {key: value for key, value in connection.items() if key not in metadata}


def _envelope(
    source: dict[str, Any], connection: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = {
        "connection_id": source["id"],
        "display_label": source["display_label"],
        "database_name": source["name"],
    }
    if connection is not None:
        result["connection"] = connection
    return result


def _persist(
    source: dict[str, Any], connection: dict[str, Any], *, migrate_inline: bool = True
) -> None:
    storage = get_connection_storage(source["id"])
    reference = storage["credential_ref"]
    if reference is not None or (migrate_inline and is_vault_configured()):
        reference = (
            reference if reference is not None else f"connections/{source['id']}"
        )
        try:
            if not is_vault_configured():
                raise ValueError
            write_secret(reference, connection)
        except Exception:
            raise SourceUnavailableError(
                "source_unavailable: credential write failed"
            ) from None
    insert_connection(
        connection_id=source["id"],
        database_name=source["name"],
        display_label=source["display_label"],
        credential_ref=reference,
        connection="" if reference is not None else json.dumps(connection),
    )


def _with_stored_secrets(
    connection: dict[str, Any], stored: dict[str, Any]
) -> dict[str, Any]:
    """Fill blank credentials from the stored connection.

    The UI never gets credentials back, so it cannot send them back either.
    Taking blank literally would mean re-typing an access token just to correct
    a hostname, and would silently wipe the credential of anyone who didn't.
    """
    merged = dict(connection)
    for field in SECRET_FIELDS:
        if not str(merged.get(field) or "").strip() and stored.get(field):
            merged[field] = stored[field]
    return merged


def _ingest_scope(connection: dict[str, Any]) -> list[str]:
    """Everything that decides *what* ingestion pulls in.

    The schema allowlist plus the table allow/deny regexes. All three change
    which relations land in the catalog, so an edit to any of them has to
    re-ingest — otherwise a newly excluded table stays in the catalog and
    retrieval keeps offering it, which is exactly what the operator used the
    pattern to prevent.
    """
    schemas = sorted(str(schema) for schema in (connection.get("schemas") or []))
    return schemas + [f"{key}={str(connection.get(key) or '')}" for key in FILTER_KEYS]


def _refresh_connection_caches() -> None:
    """Rebuild everything holding a connector or a cached connection flag.

    Each refresh is best-effort, because callers run this *after* they have
    already persisted. Letting one raise would abandon the rest of the write:
    an update re-ingests only when the schema allowlist moved, so failing here
    would skip that ingest with the new allowlist already stored, and the retry
    would compare the new list against itself and skip it again — permanently.
    A stale cache, by contrast, costs one refresh.
    """
    # Imported here rather than at module scope: ``databricks_oauth`` reaches
    # back into the server package, so a top-level import is circular.
    from auto_ontology.connectors.databricks_oauth import (
        invalidate_sso_federation_cache,
    )

    caches = (
        ("connectors", invalidate_connectors_cache),
        ("chat workers", refresh_chat_workers),
        ("SSO federation", invalidate_sso_federation_cache),
    )
    for name, refresh in caches:
        try:
            refresh()
        except Exception:
            logger.warning("cache_refresh_failed: %s", name)


def test_connection(
    connection: dict[str, Any],
    *,
    replacing_connection_id: str | None = None,
    replacing: str | None = None,
) -> list[str]:
    """Validate credentials for the settings UI test action.

    Returns the connection's schemas when the connector supports enumerating
    them (currently Databricks and Snowflake); enumerating doubles as the
    connectivity check. Connectors without schema enumeration just ``ping()``
    and return ``[]``.

    When the form names a specific ``schema``, the test also confirms that schema
    exists — the UI then skips schema selection, so a typo would otherwise only
    surface much later as an ingest that silently finds nothing.

    ``replacing_connection_id`` selects the source whose blank credentials are
    retained; ``replacing`` is the unambiguous legacy physical-name adapter.
    New connections may share a physical name.
    """
    database_name = str(connection.get("database") or "").strip()
    if not database_name:
        raise ValueError("Database name is required")

    requested_schema = str(connection.get("schema") or "").strip()
    connection = _driver_settings(connection)
    if requested_schema:
        connection["schema"] = requested_schema
    if replacing is not None or replacing_connection_id is not None:
        source = _selected_source(replacing_connection_id, replacing)
        stored = _stored_connection(source["id"])
        if database_name != source["name"]:
            raise ValueError("A connection's database name cannot be changed")
        if connection.get("type") != stored.get("type"):
            raise ValueError("A connection's type cannot be changed")
        connection = _with_stored_secrets(connection, stored)

    # Surface a bad allow/deny pattern here, while the operator is still looking
    # at the form. Left to ingestion it would be logged and skipped, and the
    # tables they meant to exclude would be ingested anyway.
    from_connection(connection)

    try:
        connection_string = build_connection_string(connection)
        connector = create_connector(connection_string)
    except Exception:
        raise SourceUnavailableError(
            "source_unavailable: connection test failed"
        ) from None
    try:
        # ``get_schemas`` runs a real query, so it validates connectivity and
        # credentials just like ``ping`` while also returning the schema list.
        get_schemas = getattr(connector, "get_schemas", None)
        if get_schemas is None:
            connector.ping()
            return []

        schemas = list(get_schemas())
    except Exception:
        raise SourceUnavailableError(
            "source_unavailable: connection test failed"
        ) from None
    finally:
        try:
            connector.close()
        except Exception:
            logger.warning("connector_close_failed")
    if requested_schema and not any(
        str(name).strip().casefold() == requested_schema.casefold() for name in schemas
    ):
        raise ValueError(
            f"Schema {requested_schema!r} was not found in {database_name!r}."
        )
    return schemas


def create_connection(
    connection: dict[str, Any],
    display_label: str | None = None,
) -> dict[str, Any]:
    """Create a UI-managed connection stored on the catalog database row."""

    # ``schema`` is the form's shorthand for a single-schema allowlist; the caller
    # turns it into ``schemas``. Storing the raw field too would leave two sources of
    # truth for what gets ingested, so drop it.
    _require_managed_mode()
    connection = _driver_settings(connection)

    database_name = connection.get("database")
    if not isinstance(database_name, str) or not database_name.strip():
        raise ValueError("Database name is required")

    # Raises InvalidTableFilterError (a ValueError) -> 422, rather than storing
    # a pattern that can never be honoured.
    from_connection(connection)

    label = _check_label(database_name if display_label is None else display_label)
    connection_id = str(uuid4())

    # With Vault configured, store credentials in Vault and keep them off the
    # catalog row; otherwise fall back to persisting the JSON on the row.
    vault_configured = is_vault_configured()
    reference = f"connections/{connection_id}" if vault_configured else None
    if vault_configured:
        try:
            write_secret(reference, connection)
        except Exception:
            raise SourceUnavailableError(
                "source_unavailable: credential write failed"
            ) from None
    try:
        insert_connection(
            connection_id=connection_id,
            display_label=label,
            credential_ref=reference,
            connection=json.dumps(connection) if reference is None else "",
            database_name=database_name,
        )
    except Exception as exc:
        if reference is not None:
            try:
                delete_secrets(reference)
            except Exception:
                logger.warning("orphan_secret: connection_id=%s", connection_id)
        if isinstance(exc, LabelConflictError):
            raise LabelConflictError("label_conflict: label already exists") from None
        raise ValueError("source_conflict: connection create failed") from None

    _refresh_connection_caches()

    trigger_ingest(connection_id)

    return _envelope(
        {"id": connection_id, "display_label": label, "name": database_name}, connection
    )


def update_connection(
    connection_id: str | None = None,
    database_name: str | None = None,
    connection: dict[str, Any] | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    """Rewrite an existing connection's settings, keeping its identity.

    The row is selected by stable ID. Physical ``database`` and driver ``type``
    cannot change. A label-only edit never loads credentials or refreshes workers.

    Re-ingest only when the schema/table filters moved. Rotating a token or fixing a
    hostname leaves the ingested graph correct, and re-ingesting it would be a
    long, surprising side effect of saving a form.
    """
    # Same shorthand the create path drops: `schema` is the single-schema form
    # of `schemas`, and storing both leaves two sources of truth.
    source = _selected_source(connection_id, database_name)
    if display_label is not None:
        source = {**source, "display_label": _check_label(display_label, source["id"])}
    if connection is None:
        update_connection_label(
            connection_id=source["id"], display_label=source["display_label"]
        )
        return _envelope(source, {})
    connection = _driver_settings(connection)
    stored = _stored_connection(source["id"])
    database_name = source["name"]

    incoming_database = str(connection.get("database") or "").strip()
    if incoming_database and incoming_database != database_name:
        raise ValueError("A connection's database name cannot be changed")

    stored_type = str(stored.get("type") or "").strip()
    incoming_type = str(connection.get("type") or "").strip()
    if incoming_type and stored_type and incoming_type != stored_type:
        raise ValueError("A connection's type cannot be changed")

    updated = _with_stored_secrets(
        {
            **connection,
            "type": stored_type or incoming_type,
            "database": database_name,
        },
        stored,
    )
    from_connection(updated)
    _persist(source, updated)

    _refresh_connection_caches()

    if _ingest_scope(updated) != _ingest_scope(stored):
        trigger_ingest(source["id"])

    return _envelope(source, updated)


def set_sso_federation(
    *, connection_id: str | None = None, database_name: str | None = None, enabled: bool
) -> dict[str, Any]:
    """Toggle "authenticate as signed-in user" on an existing connection.

    Deliberately narrow: it rewrites only the ``sso_federation`` flag on the
    stored connection, so callers never have to re-send credentials to change
    it. Ingestion is unaffected — it always uses the stored access token — so
    no re-ingest is triggered.
    """
    source = _selected_source(connection_id, database_name)
    connection = _stored_connection(source["id"])

    updated = {**connection, "sso_federation": enabled}

    _persist(source, updated, migrate_inline=False)

    # Connectors cache the credential, chat workers hold connectors, and the
    # chat endpoint caches this very flag, so all three have to be rebuilt for
    # the change to take effect on the next question rather than after a TTL.
    _refresh_connection_caches()

    return {**_envelope(source, updated), "sso_federation": enabled}


def delete_connection(
    database_name: str | None = None, *, connection_id: str | None = None
) -> dict[str, Any] | None:
    """Delete a UI-managed connection and tear down its ingested database graph.

    ``None`` when no connection carries that name, which the route turns into a
    404.

    Removing the record is done here rather than delegated to the ingestion
    service: a connection is a catalog row carrying connection metadata, and
    clearing that is a write this process can do. Delegating it meant a delete
    silently did nothing whenever the service was unreachable — and still
    answered 200, because the call to it is best-effort.

    Deleting the ingested graph stays the service's job and stays best-effort.
    It is the slow half, and failing it leaves stale data rather than a
    connection that comes back from the dead.
    """
    try:
        source = _selected_source(connection_id, database_name)
    except LookupError:
        return None

    # Keep the binding available for a retry if exact-reference deletion fails.
    reference = get_connection_storage(source["id"])["credential_ref"]
    if reference is not None:
        try:
            if not is_vault_configured():
                raise ValueError
            delete_secrets(reference)
        except Exception:
            raise SourceUnavailableError(
                "source_unavailable: credential delete failed"
            ) from None

    clear_connection(connection_id=source["id"])

    _refresh_connection_caches()

    trigger_reset(source["id"])

    return _envelope(source)
