# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Term families: the reads detection needs, and the one write that stores them.

Detection itself is in ``auto_ontology.semantic.term_families``; nothing here
decides what a family is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import (
    ColumnElement,
    Select,
    Table,
    delete,
    func,
    insert,
    literal,
    select,
    union,
)
from sqlalchemy.dialects.postgresql import aggregate_order_by, array

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.dal.source_scope import query_source_scope


def _database_schema_ids(
    database_name: str, connection_id: str | None = None
) -> Select:
    """The schemas of one database, as a subquery."""
    statement = (
        select(s.catalog_schema.c.id)
        .join(
            s.catalog_database,
            s.catalog_database.c.id == s.catalog_schema.c.database_id,
        )
        .where(s.catalog_database.c.name == database_name)
    )
    if connection_id is not None:
        statement = statement.where(s.catalog_schema.c.database_id == connection_id)
    return statement


def attribute_family_prefix(
    attribute_id: ColumnElement, *, connection_id: str | None = None
) -> ColumnElement:
    """The attribute Term's family prefix in its owning schema, or ``NULL``."""
    attribute = s.column_attribute.alias("family_attribute")
    owner = s.catalog_table.alias("family_owner")
    statement = (
        select(s.term_family.c.prefix)
        .select_from(
            s.column_attribute__term.join(
                s.term__term_family,
                s.term__term_family.c.term_id == s.column_attribute__term.c.term_id,
            )
            .join(s.term_family, s.term_family.c.id == s.term__term_family.c.family_id)
            .join(attribute, attribute.c.id == s.column_attribute__term.c.attribute_id)
            .join(owner, owner.c.id == attribute.c.table_id)
        )
        .where(
            s.column_attribute__term.c.attribute_id == attribute_id,
            s.term_family.c.schema_id == owner.c.schema_id,
        )
    )
    if connection_id is not None:
        statement = statement.where(
            s.term_family.c.schema_id.in_(
                select(s.catalog_schema.c.id).where(
                    s.catalog_schema.c.database_id == connection_id
                )
            )
        )
    return statement.order_by(s.term_family.c.prefix).limit(1).scalar_subquery()


def fetch_family_candidates(
    database_name: str, *, connection_id: str | None = None
) -> list[dict[str, Any]]:
    """One row per (table, Term) of *database_name*, with the table's columns.

    Rows carry ``table_id``, ``schema_id``, ``table_name``, ``term_id`` and
    ``columns`` — the column names in ordinal order. A table without a Term is
    absent: a family links Terms, so such a table could never be a member.
    """
    columns = (
        select(
            func.array_agg(
                aggregate_order_by(
                    s.catalog_column.c.name, s.catalog_column.c.ordinal_position
                )
            )
        )
        .where(s.catalog_column.c.table_id == s.catalog_table.c.id)
        .scalar_subquery()
    )
    statement = (
        select(
            s.catalog_table.c.id.label("table_id"),
            s.catalog_table.c.schema_id,
            s.catalog_table.c.name.label("table_name"),
            s.table__term.c.term_id,
            func.coalesce(columns, array([], type_=s.catalog_column.c.name.type)).label(
                "columns"
            ),
        )
        .join(s.table__term, s.table__term.c.table_id == s.catalog_table.c.id)
        .where(
            s.catalog_table.c.schema_id.in_(
                _database_schema_ids(database_name, connection_id)
            )
        )
        .order_by(s.catalog_table.c.name, s.table__term.c.term_id)
    )
    return [dict(row) for row in store().query_read(statement)]


def fetch_family_edges(
    database_name: str, *, connection_id: str | None = None
) -> list[dict[str, Any]]:
    """Data edges between the tables of *database_name*, column to column.

    Two sources, both data the customer brought rather than anything compiled:

    * declared foreign keys (``origin="fk"``);
    * joins observed in ingested SQL (``origin="sql"``). A join counts only when
      at least one statement it was seen in is not owned by a SqlAttribute or
      a CustomAnalysis — the same exclusion ``fetch_terms_with_sqls`` applies,
      so SQL this app generated never vouches for a join.

    Rows carry ``source_table_id``, ``source_column``, ``target_table_id``,
    ``target_column`` and ``origin``. Both ends are in *database_name*.
    """
    source = s.catalog_column.alias("source")
    target = s.catalog_column.alias("target")
    schemas = _database_schema_ids(database_name, connection_id)
    owned = union(
        select(s.sql_attribute__sql.c.sql_query_id),
        select(s.custom_analysis__sql.c.sql_query_id),
    )
    ref = func.unnest(s.column__join.c.refs).column_valued("ref")
    customer_statement = select(literal(1)).where(
        func.split_part(ref, "|", 1).notin_(owned)
    )
    if connection_id is not None:
        customer_statement = customer_statement.where(
            select(s.sql_query.c.id)
            .where(
                s.sql_query.c.id == func.split_part(ref, "|", 1),
                query_source_scope(connection_id),
            )
            .exists()
        )
    customer_sql = customer_statement.exists()

    def edges(link: Table, origin: str, *where: ColumnElement) -> Select:
        source_table = s.catalog_table.alias(f"{origin}_source_table")
        target_table = s.catalog_table.alias(f"{origin}_target_table")
        return (
            select(
                source.c.table_id.label("source_table_id"),
                source.c.name.label("source_column"),
                target.c.table_id.label("target_table_id"),
                target.c.name.label("target_column"),
                literal(origin).label("origin"),
            )
            .select_from(
                link.join(source, source.c.id == link.c.source_column_id)
                .join(target, target.c.id == link.c.target_column_id)
                .join(source_table, source_table.c.id == source.c.table_id)
                .join(target_table, target_table.c.id == target.c.table_id)
            )
            .where(
                source_table.c.schema_id.in_(schemas),
                target_table.c.schema_id.in_(schemas),
                *where,
            )
        )

    statement = union(
        edges(s.column__foreign_key, "fk"),
        edges(s.column__join, "sql", customer_sql),
    )
    return [dict(row) for row in store().query_read(statement)]


def fetch_table_families(
    database_name: str, *, connection_id: str | None = None
) -> list[dict[str, Any]]:
    """One row per family membership of a table of *database_name*.

    Rows carry ``table_id``, ``family_id``, ``prefix`` and ``base_name``,
    reached through the table's Terms. A table none of whose Terms is in a
    family is absent.
    """
    statement = (
        select(
            s.table__term.c.table_id,
            s.term__term_family.c.family_id,
            s.term_family.c.prefix,
            s.term__term_family.c.base_name,
        )
        .join(
            s.term__term_family,
            s.term__term_family.c.term_id == s.table__term.c.term_id,
        )
        .join(s.term_family, s.term_family.c.id == s.term__term_family.c.family_id)
        .join(s.catalog_table, s.catalog_table.c.id == s.table__term.c.table_id)
        .where(
            s.catalog_table.c.schema_id.in_(
                _database_schema_ids(database_name, connection_id)
            ),
            s.term_family.c.schema_id == s.catalog_table.c.schema_id,
        )
        .order_by(s.table__term.c.table_id, s.term__term_family.c.family_id)
    )
    return [dict(row) for row in store().query_read(statement)]


def replace_term_families(
    database_name: str,
    families: Iterable[tuple[str, str, Mapping[str, str]]],
    *,
    connection_id: str | None = None,
) -> int:
    """Replace every family of *database_name* with *families*.

    Each family is ``(schema_id, prefix, {term_id: base_name})``. The delete
    and the inserts share one transaction, so a reader sees the old families or
    the new ones, never a database with none. Returns the number of Terms
    linked.
    """
    linked = 0
    families = list(families)
    memberships: set[tuple[str, str]] = set()
    for schema_id, _, members in families:
        for term_id in members:
            membership = (schema_id, term_id)
            if membership in memberships:
                raise ValueError("A Term can belong to only one family per schema")
            memberships.add(membership)
    with write_transaction():
        if connection_id is not None:
            schemas = {
                row["id"]
                for row in store().query_read(
                    _database_schema_ids(database_name, connection_id)
                )
            }
            for schema_id, _, members in families:
                if schema_id not in schemas:
                    raise ValueError("Family schema does not belong to connection_id")
                valid_terms = {
                    row["term_id"]
                    for row in store().query_read(
                        select(s.table__term.c.term_id)
                        .join(
                            s.catalog_table,
                            s.catalog_table.c.id == s.table__term.c.table_id,
                        )
                        .where(
                            s.catalog_table.c.schema_id == schema_id,
                            s.table__term.c.term_id.in_(list(members)),
                        )
                    )
                }
                if set(members) != valid_terms:
                    raise ValueError("Family members do not belong to schema")
        store().query_write(
            delete(s.term_family).where(
                s.term_family.c.schema_id.in_(
                    _database_schema_ids(database_name, connection_id)
                )
            )
        )
        for schema_id, prefix, members in families:
            if not members:
                continue
            family_id = store().query_write(
                insert(s.term_family)
                .values(schema_id=schema_id, prefix=prefix)
                .returning(s.term_family.c.id)
            )[0]["id"]
            store().query_write(
                insert(s.term__term_family).values(
                    [
                        {"term_id": term_id, "family_id": family_id, "base_name": base}
                        for term_id, base in members.items()
                    ]
                )
            )
            linked += len(members)
    return linked
