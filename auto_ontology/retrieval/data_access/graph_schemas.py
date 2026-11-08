# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-backed schema and node lookups.

Builds the :class:`Schema` objects consumed by the SQL parser plus generic
``fetch_item_by_id`` helpers.

All direct store access lives in auto_ontology/dal/datasources.py.
This module only keeps the pure-Python ``get_schemas_by_ids`` that assembles
pandas DataFrames into :class:`Schema` objects.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from auto_ontology.catalog.model import CatalogNode
from auto_ontology.catalog.constants import Labels
from auto_ontology.catalog.model import Schema

from auto_ontology.dal.datasources import (
    fetch_all_schema_ids,
    fetch_schemas_by_ids,
    fetch_item_by_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_all_schema_ids",
    "fetch_item_by_id",
    "get_schemas_by_ids",
]


def get_schemas_by_ids(
    relevant_schemas_ids: list | None = None, *, connection_id: str | None = None
) -> dict:
    """Assemble one source's parser snapshot with its registered IDs.

    An empty selection stays empty. Multiple roots cannot fit the parser's
    schema-name map, even if their physical names coincide; select one first.
    """
    if not isinstance(connection_id, str) or not connection_id.strip():
        raise ValueError("source_required: schema snapshot requires connection_id")
    if relevant_schemas_ids == []:
        return {}
    before_get_all = time.time()
    data_array = fetch_schemas_by_ids(relevant_schemas_ids, connection_id=connection_id)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    if not data_array:
        return {}
    data_df = pd.DataFrame(data_array)
    if set(data_df["connection_id"]) != {connection_id}:
        raise ValueError("source_required: schema snapshot must belong to one source")
    database_name = data_df.iloc[0]["database_name"]
    database_node = CatalogNode(
        name=database_name,
        label=Labels.DB,
        props={"name": database_name},
        existing_id=connection_id,
        match_props={"id": connection_id},
    )
    all_schemas = {}
    for schema_id, rows in data_df.groupby("schema_id", sort=False):
        schema_name = rows.iloc[0]["table_schema"]
        tables = (
            rows[["database_name", "table_schema", "table_name", "table_id"]]
            .drop_duplicates(subset=["table_id"])
            .rename(columns={"table_id": "id"})
        )
        columns = rows.rename(columns={"column_id": "id"})
        schema = Schema(database_node, tables, columns, is_creation_mode=False)
        schema.create_schema_node(schema_name, id=schema_id)
        all_schemas[schema_name.lower()] = schema
    logger.info(f"total time for get_schemas_by_ids(): {time.time() - before_get_all}")
    return all_schemas
