# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve active source IDs before loading and caching their connectors."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

from auto_ontology.catalog.table_filter import TableFilter, from_connection
from auto_ontology.connectors import sources
from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.connectors.connection_string_factory import build_connection_string
from auto_ontology.connectors.clickhouse import ClickHouseDatabase
from auto_ontology.connectors.databricks import DatabricksDatabase
from auto_ontology.connectors.duckdb import DuckDBDatabase
from auto_ontology.connectors.heavydb import HeavyDBDatabase
from auto_ontology.connectors.kyuubi import KyuubiDatabase
from auto_ontology.connectors.mysql import MySQLDatabase
from auto_ontology.connectors.postgres import PostgresDatabase
from auto_ontology.connectors.snowflake import SnowflakeDatabase
from auto_ontology.connectors.sqlite import SQLiteDatabase
from auto_ontology.connectors.trino import TrinoDatabase

logger = logging.getLogger(__name__)

# Every connector here MUST expose a ``dialect`` that sqlglot recognises
# (one of ``sqlglot.dialects.DIALECTS``). Callers pass ``connector.dialect``
# straight into ``sqlglot.parse_one(read=...)`` / ``Expression.sql(dialect=...)``
# with no translation layer in between, and sqlglot raises on an unknown
# dialect *name* before it ever looks at the SQL — so a bad name is a hard
# error, not a silent fall back to the generic parser.
#
# An engine sqlglot has no dialect for must map itself onto the closest one it
# does have; see ``HeavyDBDatabase.dialect``, which reports ``"postgres"``.
# ``test_registry_dialects.py`` enforces this.
#
# Note the keys below are connection-string schemes, NOT dialects: ``postgresql``,
# ``heavydb`` and ``kyuubi`` are valid keys but none is a sqlglot dialect name.
CONNECTOR_REGISTRY: dict[str, type[SQLDatabase]] = {
    "postgres": PostgresDatabase,
    "postgresql": PostgresDatabase,
    "mysql": MySQLDatabase,
    "databricks": DatabricksDatabase,
    "duckdb": DuckDBDatabase,
    "snowflake": SnowflakeDatabase,
    "clickhouse": ClickHouseDatabase,
    "heavydb": HeavyDBDatabase,
    "kyuubi": KyuubiDatabase,
    "sqlite": SQLiteDatabase,
    "trino": TrinoDatabase,
}

_connectors: dict[str, SQLDatabase] = {}
_connection_specs: dict[str, tuple[str | None, dict]] = {}


def parse_connection_strings(raw: str) -> list[str]:
    """Return connection URLs separated by commas, newlines, or both."""
    return [
        connection_string.strip()
        for connection_string in re.split(r"[,\r\n]+", raw)
        if connection_string.strip()
    ]


def _redact(connection_string: str) -> str:
    """Mask the password/token in a connection string before logging it.

    Connection strings carry a PAT (or, with per-user auth, a caller's exchanged
    Databricks token), so the raw value must never reach the logs.

    An uploaded Kyuubi truststore is also collapsed: it is base64 keystore bytes
    and runs to hundreds of kilobytes, which would otherwise be emitted verbatim
    on every connector failure.
    """
    connection_string = re.sub(
        r"(truststore_data=)[^&]+", r"\1<keystore>", connection_string
    )
    parsed = urlparse(connection_string)
    if not parsed.password:
        return connection_string
    return connection_string.replace(parsed.password, "***", 1)


def _schema_filter(connection: dict) -> list[str] | None:
    """Clean optional ``schemas`` ingestion allowlist off a connection dict.

    Returns a list of non-empty schema names, or ``None`` when absent (meaning
    "all schemas"). Only honoured by connectors that support schema filtering.
    """
    raw = connection.get("schemas")
    if not isinstance(raw, list):
        return None
    schemas = [str(s).strip() for s in raw if str(s).strip()]
    return schemas or None


def _table_filter(connection: dict) -> TableFilter:
    """Build the regex table allow/deny filter off a connection dict.

    A malformed pattern is logged and dropped here rather than raised: this runs
    while loading every configured connection, and one bad pattern must not stop
    the others from loading. The API validates patterns on save (422), so an
    invalid one should never reach this point.
    """
    try:
        return from_connection(connection)
    except ValueError:
        logger.exception(
            "Ignoring invalid table filter on connection %r",
            connection.get("database"),
        )
        return TableFilter()


def invalidate_connectors_cache() -> None:
    """Drop cached connectors so the next :func:`get_connectors` reloads."""
    for connection_id, connector in _connectors.items():
        try:
            connector.close()
        except Exception:
            logger.warning("Failed to close connector %s", connection_id)
    _connectors.clear()
    _connection_specs.clear()


def create_connector(
    connection_string: str,
    schemas: list[str] | None = None,
    table_filter: "TableFilter | None" = None,
) -> SQLDatabase:
    """Parse *connection_string*, select a connector class, and return an instance.

    *schemas* is an optional ingestion allowlist. It is only honoured by
    connectors that support schema filtering (currently Databricks, Snowflake,
    Kyuubi, and Trino); for others it is ignored so their behaviour is unchanged.

    *table_filter* is the connection's regex allow/deny pair. Unlike *schemas*
    it applies to every connector, because it is enforced centrally in
    :func:`auto_ontology.catalog.extract.create_dataframe` rather than pushed
    down into each driver's introspection query.
    """
    try:
        parsed = urlparse(connection_string)
        scheme = parsed.scheme

        if not scheme:
            raise ValueError("Invalid connection string: No protocol scheme found.")

        connector_type = scheme.split("+", 1)[0].lower()
        connector_class = CONNECTOR_REGISTRY.get(connector_type)

        if connector_class is None:
            raise ValueError("Unsupported database type")

        if schemas and connector_class in (
            DatabricksDatabase,
            SnowflakeDatabase,
            KyuubiDatabase,
            TrinoDatabase,
        ):
            connector = connector_class(connection_string, schemas=schemas)
        else:
            connector = connector_class(connection_string)

        # Carried on the instance rather than passed to __init__: every
        # connector must honour it, and none of them should have to accept a
        # new constructor argument to do so.
        if table_filter is not None and table_filter.active:
            connector.table_filter = table_filter
        return connector

    except Exception:
        raise ValueError("source_unavailable: could not initialize connector") from None


def _resolve_source(connection_id: str | None) -> sources.SourceDescriptor:
    active = sources.list_active_sources()
    if connection_id is None:
        if len(active) > 1:
            raise ValueError("source_required: select a connection ID")
        if len(active) == 1:
            return active[0]
    else:
        for source in active:
            if source.connection_id == connection_id:
                return source
    raise ValueError("source_unavailable: selected source is not active")


def _load_spec(source: sources.SourceDescriptor) -> tuple[str | None, dict]:
    """Read secrets only after the active source context has been validated."""
    from auto_ontology.dal.connections import load_source_connection

    try:
        if source.configuration_origin == "managed":
            return None, load_source_connection(source.connection_id)
        desired = sources.parse_env_sources(os.environ.get("CONNECTION_STRINGS", ""))
        matches = [
            item
            for item in desired
            if item.connection_id == source.connection_id
            or (
                item.connection_id is None
                and item.database_name == source.database_name
            )
        ]
        if len(matches) == 1:
            return matches[0]._driver_url, {}
    except Exception:
        pass
    raise ValueError("source_unavailable: selected source configuration unavailable")


def _create_source_connector(
    source: sources.SourceDescriptor, spec: tuple[str | None, dict]
) -> SQLDatabase:
    url, settings = spec
    try:
        connector = create_connector(
            url if url is not None else build_connection_string(settings),
            schemas=_schema_filter(settings),
            table_filter=_table_filter(settings),
        )
        connector.connection_id = source.connection_id
        return connector
    except Exception:
        raise ValueError(
            "source_unavailable: could not initialize selected source"
        ) from None


def _cached_connector(
    source: sources.SourceDescriptor, spec: tuple[str | None, dict] | None = None
) -> SQLDatabase:
    source_id = source.connection_id
    if source_id not in _connectors:
        spec = spec if spec is not None else _load_spec(source)
        connector = _create_source_connector(source, spec)
        _connectors[source_id] = connector
        _connection_specs[source_id] = spec
    return _connectors[source_id]


def get_connector(connection_id: str) -> SQLDatabase:
    """Return this active source's cached stored-credential connector."""
    return _cached_connector(_resolve_source(connection_id))


def get_connectors_for_subject_token(
    subject_token: str | None, connection_id: str | None = None
) -> list[SQLDatabase]:
    """Return one selected connector; federated callers never use the stored PAT.

    Omitting the ID is supported only with one active descriptor. Per-user
    connectors are fresh and never enter the service-credential cache.
    """
    from auto_ontology.connectors.databricks_oauth import (
        DatabricksOAuthError,
        exchange_subject_token,
        uses_sso_federation,
    )

    source = _resolve_source(connection_id)
    spec = _connection_specs.get(source.connection_id)
    if spec is None:
        spec = _load_spec(source)
    _, settings = spec
    if not uses_sso_federation(settings):
        return [_cached_connector(source, spec)]
    if not subject_token:
        raise DatabricksOAuthError(
            "Selected source requires a signed-in user's SSO token"
        )
    try:
        access_token = exchange_subject_token(
            str(settings.get("host") or ""), subject_token
        )
        if not access_token:
            raise DatabricksOAuthError
    except Exception:
        raise DatabricksOAuthError(
            "Selected source SSO token exchange failed"
        ) from None
    return [
        _create_source_connector(
            source, (None, {**settings, "access_token_override": access_token})
        )
    ]


def get_connectors() -> list[SQLDatabase]:
    """Explicit all-source ingestion enumeration, skipping unavailable sources."""
    loaded: list[SQLDatabase] = []
    for source in sources.list_active_sources():
        try:
            loaded.append(_cached_connector(source))
        except Exception:
            logger.warning(
                "Source %s unavailable; skipping connector", source.connection_id
            )
    return loaded
