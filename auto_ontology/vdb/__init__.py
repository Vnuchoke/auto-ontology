# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDB initialization — data layer and semantic layer collections."""

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB

DATA_VDB_COLLECTION: str = "data_objects_layer"
SEMANTIC_VDB_COLLECTION: str = "semantic_layer"
VDB_SCHEMA: str = "vdb"


def get_vdb(
    *,
    database_name: str | None = None,
    connection_id: str | None = None,
    collection_name: str = DATA_VDB_COLLECTION,
    reset: bool = False,
) -> PostgresVDB:
    """Build a PostgresVDB pointed at the local pgvector-enabled Postgres.

    ``connection_id`` scopes writes/search/reset; ``database_name`` is physical context only.
    """
    connection_string = get_postgres_connection_string()
    kwargs: dict = {
        "connection_string": connection_string,
        "collection_name": collection_name,
        "schema_name": VDB_SCHEMA,
        "connection_id": connection_id,
    }
    if database_name:
        kwargs["database_name"] = database_name
    return PostgresVDB(reset=reset, **kwargs)


def get_data_vdb(
    *,
    database_name: str | None = None,
    connection_id: str | None = None,
    reset: bool = False,
) -> PostgresVDB:
    """Backward-compatible alias for :func:`get_vdb` (tabular / data layer)."""
    return get_vdb(
        database_name=database_name, connection_id=connection_id, reset=reset
    )


def get_semantic_vdb(
    *,
    database_name: str | None = None,
    connection_id: str | None = None,
    reset: bool = False,
) -> PostgresVDB:
    """Build a PostgresVDB for the semantic layer."""
    return get_vdb(
        database_name=database_name,
        connection_id=connection_id,
        collection_name=SEMANTIC_VDB_COLLECTION,
        reset=reset,
    )


def get_global_pql_vdb() -> PostgresVDB:
    """Explicit global PQL vocabulary client; never a generic unscoped fallback."""
    return PostgresVDB(
        connection_string=get_postgres_connection_string(),
        collection_name=SEMANTIC_VDB_COLLECTION,
        schema_name=VDB_SCHEMA,
        global_pql=True,
    )
