# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared SQL validation helpers used by both CustomAnalysis and SqlAttribute DALs."""

from __future__ import annotations

from typing import Any

from auto_ontology.catalog.sql_parse import parse_query_single

from sqlalchemy import select
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store
from auto_ontology.dal.datasources import fetch_schema_ids_for_connection
from auto_ontology.retrieval.data_access.graph_schemas import (
    get_schemas_by_ids,
)

# sqlglot's default dialect is the generic ANSI parser, so try it before
# falling back to postgres.
_DEFAULT_DIALECTS = ["", "postgres"]


class SqlParseError(Exception):
    """SQL could not be parsed or resolved against the catalog."""


def get_dialects(connection_id: str) -> list[str]:
    """Read only the selected root's dialect; never open live connectors."""
    rows = store().query_read(
        select(s.catalog_database.c.dialect).where(
            s.catalog_database.c.id == connection_id
        )
    )
    return (
        [rows[0]["dialect"]] if rows and rows[0]["dialect"] else list(_DEFAULT_DIALECTS)
    )


def get_schemas(connection_id: str) -> dict:
    """Load selected catalog metadata, including roots without live connections."""
    scoped_ids = fetch_schema_ids_for_connection(connection_id)
    if not scoped_ids:
        return {}
    return get_schemas_by_ids(scoped_ids, connection_id=connection_id)


def validate_sql(
    sql: str, dialects: list[str], schemas: dict, *, connection_id: str
) -> Any:
    """Parse ``sql`` against the catalog. Returns query_obj or raises.

    Raises :class:`SqlParseError` when the SQL can't be parsed
    (sqlglot syntax / dialect error) or doesn't resolve to any
    table known to the catalog.
    """
    try:
        query_obj = parse_query_single(
            sql=sql, dialects=dialects, schemas=schemas, connection_id=connection_id
        )
    except Exception as exc:
        raise SqlParseError(f"SQL parse error: {exc}") from exc

    if query_obj is None:
        raise SqlParseError(
            "SQL doesn't reference any table known to the catalog; "
            "ingest the schema first or check the query",
        )
    return query_obj
