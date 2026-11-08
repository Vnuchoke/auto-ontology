# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI-managed database connections.

Managed metadata lives on its stable catalog root ID. Physical names may repeat;
credential references are explicit and never resolved from a database name.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import func, null, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from auto_ontology.connectors.vault import read_secret
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction

logger = logging.getLogger(__name__)


class LabelConflictError(ValueError):
    """A label conflicts under the database's normalization rules."""


def _write_connection(statement: Any) -> list[dict[str, Any]]:
    try:
        return store().query_write(statement)
    except IntegrityError as exc:
        if getattr(getattr(exc.orig, "diag", None), "constraint_name", None) == (
            "uq_catalog_database_display_label_lower"
        ):
            raise LabelConflictError("label_conflict: label already exists") from None
        raise ValueError("source_conflict: connection write failed") from None
    except Exception:
        raise ValueError("source_unavailable: connection write failed") from None


def list_source_metadata() -> list[dict[str, Any]]:
    """Read root identity only, including inactive roots; never load credentials."""
    root = s.catalog_database.c
    return store().query_read(
        select(
            root.id,
            root.name,
            root.display_label,
            root.configuration_origin,
            root.dialect,
            func.lower(func.btrim(root.display_label, " ")).label("normalized_label"),
        ).order_by(root.id)
    )


def load_source_connection(connection_id: str) -> dict[str, Any]:
    """Load only this managed source's settings; an explicit ref never falls back."""
    try:
        root = s.catalog_database.c
        rows = store().query_read(
            select(root.connection, root.credential_ref).where(
                (root.id == connection_id) & (root.configuration_origin == "managed")
            )
        )
        if not rows:
            raise ValueError
        row = rows[0]
        payload = (
            read_secret(row["credential_ref"])
            if row["credential_ref"] is not None
            else row["connection"]
        )
        if isinstance(payload, str):
            payload = json.loads(payload)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("type"), str)
            or not payload["type"].strip()
        ):
            raise ValueError
        return {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "id",
                "name",
                "connection_id",
                "database_name",
                "display_label",
                "configuration_origin",
                "credential_ref",
                "dialect",
                "normalized_label",
                "metadata_connection_id",
                "metadata_label",
            }
        }
    except Exception:
        raise ValueError(
            "source_unavailable: selected source credentials unavailable"
        ) from None


def normalize_source_labels(labels: list[str]) -> list[str]:
    """Use the same PostgreSQL normalization as the unique root-label index."""
    return [
        row["label"]
        for row in store().query_read(
            "SELECT lower(TRIM(BOTH FROM value)) AS label FROM unnest(CAST(:labels AS text[])) AS value",
            {"labels": labels},
        )
    ]


def lock_source_registration() -> None:
    """Serialize startup registration inside the caller's write transaction."""
    store().query_write("SELECT pg_advisory_xact_lock(719302112097)")


def assert_connection_id_available(connection_id: str) -> None:
    """Reject retired IDs; writers must hold the registration lock through write."""
    if store().query_read(
        select(s.retired_connection.c.connection_id).where(
            s.retired_connection.c.connection_id == connection_id
        )
    ):
        raise ValueError("source_retired: connection_id cannot be reused")


def finalize_connection_removal(connection_id: str) -> None:
    """Atomically retire a detached root, including an already missing root."""
    with write_transaction():
        lock_source_registration()
        root = s.catalog_database.c
        rows = store().query_read(
            select(
                (
                    root.configuration_origin.is_not(None)
                    | root.connection.is_not(None)
                    | root.credential_ref.is_not(None)
                ).label("configured")
            )
            .where(root.id == connection_id)
            .with_for_update()
        )
        if rows and rows[0]["configured"]:
            raise ValueError("source_conflict: source must be detached before removal")
        store().query_write(
            insert(s.retired_connection)
            .values(connection_id=connection_id)
            .on_conflict_do_nothing()
        )
        store().query_write(s.catalog_database.delete().where(root.id == connection_id))


def register_env_source(
    *, connection_id: str, display_label: str, database_name: str
) -> None:
    """Persist validated env identity only; never credentials or physical renames."""
    statement = insert(s.catalog_database).values(
        id=connection_id,
        name=database_name,
        display_label=display_label,
        configuration_origin="env",
    )
    with write_transaction():
        lock_source_registration()
        assert_connection_id_available(connection_id)
        rows = store().query_write(
            statement.on_conflict_do_update(
                index_elements=[s.catalog_database.c.id],
                set_={"display_label": statement.excluded.display_label},
                where=(s.catalog_database.c.configuration_origin == "env")
                & (s.catalog_database.c.name == database_name),
            ).returning(s.catalog_database.c.id)
        )
    if not rows:
        raise ValueError(
            "source_conflict: source ownership changed during registration"
        )


def list_connections() -> list[dict[str, Any]]:
    """Resolve managed records by explicit ID/ref, never by physical name."""
    return [
        {
            "connection_id": row["id"],
            "display_label": row["display_label"],
            "database_name": row["name"],
            "connection": load_source_connection(row["id"]),
        }
        for row in list_source_metadata()
        if row["configuration_origin"] == "managed"
    ]


def get_connection_storage(connection_id: str) -> dict[str, Any]:
    """Selected managed row only; used to preserve its exact storage binding."""
    root = s.catalog_database.c
    try:
        rows = store().query_read(
            select(root.connection, root.credential_ref).where(
                (root.id == connection_id) & (root.configuration_origin == "managed")
            )
        )
    except Exception:
        raise ValueError("source_unavailable: credential binding unavailable") from None
    if not rows:
        raise LookupError("source_not_found")
    return dict(rows[0])


def update_connection_label(*, connection_id: str, display_label: str) -> None:
    """Rename metadata without reading or rewriting credentials."""
    root = s.catalog_database.c
    rows = _write_connection(
        update(s.catalog_database)
        .where((root.id == connection_id) & (root.configuration_origin == "managed"))
        .values(display_label=display_label)
        .returning(root.id)
    )
    if not rows:
        raise LookupError("source_not_found")


def insert_connection(
    *,
    connection_id: str,
    display_label: str,
    connection: str,
    database_name: str,
    credential_ref: str | None = None,
) -> dict[str, Any]:
    """Insert an exact ID or update that managed ID with the same physical name."""
    # An empty string means "no connection JSON on the row" -- the Vault path
    # writes the credentials to Vault and deliberately keeps them off the
    # catalog. `json.loads("")` raises, and because the Vault write happens
    # first, the secret was already stored when the insert 500s: no
    # catalog_database row, so the connection is invisible to list_connections
    # and cannot be ingested.
    payload = json.loads(connection) if connection and connection.strip() else None
    root = s.catalog_database.c
    statement = insert(s.catalog_database).values(
        id=connection_id,
        name=database_name,
        display_label=display_label,
        configuration_origin="managed",
        connection=payload if payload is not None else null(),
        credential_ref=credential_ref,
    )
    with write_transaction():
        lock_source_registration()
        assert_connection_id_available(connection_id)
        rows = _write_connection(
            statement.on_conflict_do_update(
                index_elements=[root.id],
                set_={
                    "connection": statement.excluded.connection,
                    "display_label": statement.excluded.display_label,
                    "credential_ref": statement.excluded.credential_ref,
                },
                where=(root.configuration_origin == "managed")
                & (root.name == database_name),
            ).returning(s.catalog_database)
        )
    if not rows:
        raise ValueError("source_conflict: source ownership changed")
    return dict(rows[0])


def clear_connection(*, connection_id: str) -> None:
    """Detach connection metadata from the catalog database.

    Clear only this managed ID's active ownership and credential binding.

    The row itself stays. ``catalog_schema`` cascades off it, so deleting it
    here would drop the ingested catalog while leaving behind the semantic rows
    and pgvector embeddings that are identified *through* it — the half-state
    :func:`auto_ontology.dal.reset.delete_all_data` orders its own deletes to avoid.
    Tearing the graph down is that function's job.
    """
    _write_connection(
        update(s.catalog_database)
        .where(
            (s.catalog_database.c.id == connection_id)
            & (s.catalog_database.c.configuration_origin == "managed")
        )
        .values(connection=null(), credential_ref=None, configuration_origin=None)
    )


def verify_connectivity() -> None:
    """Probe the store. Raises if it is unreachable.

    Backs the health endpoint, so it must fail rather than report healthy — the
    ``SELECT 1`` is deliberately the cheapest statement that still proves a
    connection can be checked out of the pool and used.
    """
    store().query_read("SELECT 1")
