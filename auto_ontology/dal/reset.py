# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-ID resets preserve identity and global vocabulary.

Metadata commits before vector cleanup (a separate connection). Failures propagate;
retry with the captured opaque ID, never a physical name or display label.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, delete, select, union

from auto_ontology.dal import schema as s
from auto_ontology.dal.connections import finalize_connection_removal
from auto_ontology.dal.session import store, write_transaction
from auto_ontology.vdb import get_data_vdb, get_semantic_vdb


@dataclass
class ResetResult:
    """Embedding counts; the physical name is informational, never a selector."""

    database_name: str | None
    data_rows: int
    semantic_rows: int
    connection_id: str | None = None


def _connection_ids(connection_id: str | None) -> list[str]:
    if connection_id is not None:
        return [connection_id]
    return [row["id"] for row in store().query_read(select(s.catalog_database.c.id))]


def _table_ids(connection_id: str) -> Select:
    return (
        select(s.catalog_table.c.id)
        .select_from(s.catalog_table.join(s.catalog_schema))
        .where(s.catalog_schema.c.database_id == connection_id)
    )


def _delete_scoped_semantic(connection_id: str) -> int:
    """Remove owned semantics, retaining every Term with a surviving dependency."""
    tables = _table_ids(connection_id)
    attributes = select(s.column_attribute.c.id).where(
        s.column_attribute.c.table_id.in_(tables)
    )
    # Capture before deleting owners and their cascading links.
    candidates = [
        row["term_id"]
        for row in store().query_read(
            union(
                select(s.table__term.c.term_id).where(
                    s.table__term.c.table_id.in_(tables)
                ),
                select(s.column_attribute__term.c.term_id).where(
                    s.column_attribute__term.c.attribute_id.in_(attributes)
                ),
                select(s.sql_attribute__term.c.term_id)
                .select_from(s.sql_attribute__term.join(s.sql_attribute))
                .where(s.sql_attribute.c.connection_id == connection_id),
            )
        )
    ]
    deleted = len(
        store().query_write(
            delete(s.column_attribute)
            .where(s.column_attribute.c.id.in_(attributes))
            .returning(s.column_attribute.c.id)
        )
    )
    for table in (s.sql_attribute, s.custom_analysis):
        deleted += len(
            store().query_write(
                delete(table)
                .where(table.c.connection_id == connection_id)
                .returning(table.c.id)
            )
        )
    store().query_write(
        delete(s.table__term).where(s.table__term.c.table_id.in_(tables))
    )
    needed = union(
        select(s.table__term.c.term_id),
        select(s.column_attribute__term.c.term_id),
        select(s.sql_attribute__term.c.term_id),
        select(s.tag_target.c.term_id).where(s.tag_target.c.term_id.is_not(None)),
    )
    deleted += len(
        store().query_write(
            delete(s.term)
            .where(s.term.c.id.in_(candidates), s.term.c.id.not_in(needed))
            .returning(s.term.c.id)
        )
    )
    # Keep ingestion history for rebuilding semantics; remove only unlinked SQL.
    _delete_queries(connection_id)
    return deleted


def _delete_queries(connection_id: str) -> None:
    referenced = union(
        select(s.sql_query__table.c.sql_query_id),
        select(s.sql_query__column.c.sql_query_id),
        select(s.sql_attribute__sql.c.sql_query_id),
        select(s.custom_analysis__sql.c.sql_query_id),
    )
    store().query_write(
        delete(s.sql_query).where(
            s.sql_query.c.connection_id == connection_id,
            s.sql_query.c.id.not_in(referenced),
        )
    )


def _delete_data_rows(connection_id: str) -> None:
    store().query_write(
        delete(s.catalog_schema).where(s.catalog_schema.c.database_id == connection_id)
    )
    _delete_queries(connection_id)


def _delete_vectors(connection_id: str, *, semantic: bool) -> int:
    """Unbound clients work even after retirement or deletion of the root."""
    client = get_semantic_vdb() if semantic else get_data_vdb()
    try:
        return len(client.delete_by_connection(connection_id))
    finally:
        client.close()


def delete_semantic_layer(connection_id: str | None = None) -> int:
    """Reset scoped semantics/executable SQL and vectors; keep the catalog/PQL."""
    ids = _connection_ids(connection_id)
    with write_transaction():
        for cid in ids:
            _delete_scoped_semantic(cid)
    return sum(_delete_vectors(cid, semantic=True) for cid in ids)


def delete_data_layer(connection_id: str | None = None) -> int:
    """Reset child catalog and unlinked SQL; retain roots and semantic definitions."""
    ids = _connection_ids(connection_id)
    with write_transaction():
        for cid in ids:
            _delete_data_rows(cid)
    return sum(_delete_vectors(cid, semantic=False) for cid in ids)


def delete_all_data(connection_id: str | None = None) -> ResetResult:
    """Reset both layers by captured source IDs, preserving roots and global PQL."""
    ids = _connection_ids(connection_id)
    roots = (
        store().query_read(
            select(s.catalog_database.c.name).where(
                s.catalog_database.c.id == connection_id
            )
        )
        if connection_id is not None
        else []
    )
    with write_transaction():
        for cid in ids:
            _delete_scoped_semantic(cid)
            _delete_data_rows(cid)
    semantic_rows = sum(_delete_vectors(cid, semantic=True) for cid in ids)
    data_rows = sum(_delete_vectors(cid, semantic=False) for cid in ids)
    return ResetResult(
        database_name=roots[0]["name"] if roots else None,
        connection_id=connection_id,
        data_rows=data_rows,
        semantic_rows=semantic_rows,
    )


def remove_connection_data(connection_id: str) -> ResetResult:
    """Finish retired-source cleanup; release its label only after vectors succeed.

    A missing root is retryable. A remaining root must have no origin, credential
    reference or inline connection: retirement belongs to the lifecycle boundary.
    """
    with write_transaction():
        roots = store().query_read(
            select(
                (
                    s.catalog_database.c.configuration_origin.is_not(None)
                    | s.catalog_database.c.credential_ref.is_not(None)
                    | s.catalog_database.c.connection.is_not(None)
                ).label("configured")
            )
            .where(s.catalog_database.c.id == connection_id)
            .with_for_update()
        )
        if roots and roots[0]["configured"]:
            raise ValueError("Connection must be retired before final cleanup")
    result = delete_all_data(connection_id)
    finalize_connection_removal(connection_id)
    return result
