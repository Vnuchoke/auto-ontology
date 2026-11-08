# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All-or-nothing ownership predicates for executable SQL context."""

from sqlalchemy import ColumnElement, Table, and_, literal, select

from auto_ontology.dal import schema as s


def query_source_scope(connection_id: str) -> ColumnElement:
    """A statement and every linked table/column must belong to this source."""
    foreign_tables = (
        select(literal(1))
        .select_from(s.sql_query__table.join(s.catalog_table).join(s.catalog_schema))
        .where(
            s.sql_query__table.c.sql_query_id == s.sql_query.c.id,
            s.catalog_schema.c.database_id != connection_id,
        )
        .correlate(s.sql_query)
        .exists()
    )
    foreign_columns = (
        select(literal(1))
        .select_from(
            s.sql_query__column.join(s.catalog_column)
            .join(s.catalog_table)
            .join(s.catalog_schema)
        )
        .where(
            s.sql_query__column.c.sql_query_id == s.sql_query.c.id,
            s.catalog_schema.c.database_id != connection_id,
        )
        .correlate(s.sql_query)
        .exists()
    )
    return and_(
        s.sql_query.c.connection_id == connection_id, ~foreign_tables, ~foreign_columns
    )


def definition_source_scope(
    owner: Table, link: Table, owner_key: str, connection_id: str
) -> ColumnElement:
    """Do not render a partial definition when one statement has foreign ownership."""
    invalid = (
        select(literal(1))
        .select_from(link.join(s.sql_query))
        .where(
            link.c[owner_key] == owner.c.id,
            query_source_scope(connection_id).is_not(True),
        )
        .correlate(owner)
        .exists()
    )
    return and_(owner.c.connection_id == connection_id, ~invalid)
