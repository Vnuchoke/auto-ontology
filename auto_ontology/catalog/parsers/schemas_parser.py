# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn tables/columns DataFrames into Schema aggregates."""

import pandas as pd
from auto_ontology.catalog.constants import Labels
from auto_ontology.catalog.model.node import CatalogNode
from auto_ontology.catalog.model.schema import Schema
from auto_ontology.catalog.store.schemas import get_table_ids, get_column_ids
from auto_ontology.catalog.store.rows import resolve_id
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)


def parse_df(
    tables_df: pd.DataFrame,
    columns_df: pd.DataFrame,
    database_name: str,
    database_node: CatalogNode | None = None,
    *,
    connection_id: str,
) -> tuple[dict[str, Schema], CatalogNode]:
    """
    Every schema manager assumes a single database in the input file
    :param tables_df: DataFrame with columns: table_schema, table_name
    :param columns_df: DataFrame with columns: table_schema, table_name, column_name, ...
    :param database_name: name of the database, taken from the connector
    :param database_node: optional existing database node
    Assumption: the file contains schemas of a single database
    ``connection_id`` must identify a previously registered root whose physical
    name agrees with the connector. Ingest never creates or rebinds a source.
    :return:
    """
    roots = store().query_read(
        select(s.catalog_database.c.name).where(
            s.catalog_database.c.id == connection_id
        )
    )
    if not roots or roots[0]["name"] != database_name:
        raise ValueError(
            "Catalog connection_id is not registered with this database name"
        )
    if database_node is not None and (
        database_node.get_id() != connection_id
        or database_node.get_name() != database_name
    ):
        raise ValueError("Database node conflicts with catalog source context")
    if database_node is None:
        database_node = CatalogNode(
            name=database_name,
            label=Labels.DB,
            props={"name": database_name},
            existing_id=connection_id,
            match_props={"id": connection_id},
        )

    tables_df = get_table_ids(tables_df, connection_id)
    columns_df = get_column_ids(columns_df, connection_id)

    unique_schema_names = tables_df["table_schema"].unique()
    schemas = {}

    for schema_name in unique_schema_names:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == schema_name]
        schema_columns_df = columns_df.loc[columns_df["table_schema"] == schema_name]
        logger.info(f"Started parsing schema {schema_name}.")
        schema = Schema(database_node, schema_tables_df, schema_columns_df)
        schema.create_schema_node(
            schema_name,
            id=resolve_id(
                Labels.SCHEMA, {"connection_id": connection_id, "name": schema_name}
            ),
        )
        schemas.update({schema.get_schema_name().lower(): schema})
        logger.info(f"Finished parsing schema {schema.get_schema_name()}.")

    return schemas, database_node
