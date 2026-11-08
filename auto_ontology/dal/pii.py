# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence operations for automatic PII classification."""

from __future__ import annotations

from sqlalchemy import Select, Text, exists, literal, not_, select
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import IN_QUERY_BATCH, store, write_transaction
from auto_ontology.dal.source_scope import definition_source_scope
from auto_ontology.dal.tags import (
    PII_TAG_NAME,
    TARGET_COLUMN,
    TARGET_COLUMN_ATTRIBUTE,
    TARGET_SQL_ATTRIBUTE,
    detach_tag,
    get_tag_by_name,
)


def mark_columns_pii_processed(column_ids: list[str]) -> int:
    """Mark successfully classified catalog columns as processed.

    Ids are updated in batches of :data:`IN_QUERY_BATCH`. A first ingest of a
    large catalog can have more columns than psycopg's 65535 bind parameters,
    and a single ``IN`` of that many literals would raise -- silently, since
    ingest swallows a detection failure, and permanently, since nothing would
    then be marked processed to shrink the next pass.
    """

    if not column_ids:
        return 0

    marked = 0
    for offset in range(0, len(column_ids), IN_QUERY_BATCH):
        chunk = column_ids[offset : offset + IN_QUERY_BATCH]
        rows = store().query_write(
            s.catalog_column.update()
            .where(
                s.catalog_column.c.id.in_(chunk),
                s.catalog_column.c.pii_processed.is_(False),
            )
            .values(pii_processed=True)
            .returning(s.catalog_column.c.id)
        )
        marked += len(rows)
    return marked


def _column_attributes_of_tagged_columns(
    tag_id: str, connection_id: str | None = None
) -> Select:
    """Ids of unprocessed ColumnAttributes whose column carries *tag_id*.

    ``column__has_attribute`` only: that edge says the column *is* an instance of
    the attribute. ``column__semantic_fk`` says the column merely references an
    attribute describing a column elsewhere, so a PII foreign key must not make
    the thing it points at PII.

    Joined through the Term link because an attribute with no Term is invisible
    app-wide and cannot be tagged through ``attach_tag`` either.

    ``pii_processed`` is filtered here, so an attribute handled once is never a
    candidate again -- which is what lets a person remove its tag for good.
    """
    statement = (
        select(s.column__has_attribute.c.attribute_id.label("id"))
        .select_from(
            s.tag_target.join(
                s.column__has_attribute,
                s.column__has_attribute.c.column_id == s.tag_target.c.column_id,
            )
            .join(
                s.column_attribute__term,
                s.column_attribute__term.c.attribute_id
                == s.column__has_attribute.c.attribute_id,
            )
            .join(
                s.column_attribute,
                s.column_attribute.c.id == s.column__has_attribute.c.attribute_id,
            )
        )
        .where(
            s.tag_target.c.tag_id == tag_id,
            s.column_attribute.c.pii_processed.is_(False),
        )
        .distinct()
    )
    if connection_id is not None:
        statement = (
            statement.join(
                s.catalog_column,
                s.catalog_column.c.id == s.column__has_attribute.c.column_id,
            )
            .join(s.catalog_table, s.catalog_table.c.id == s.catalog_column.c.table_id)
            .join(
                s.catalog_schema, s.catalog_schema.c.id == s.catalog_table.c.schema_id
            )
            .where(
                s.catalog_schema.c.database_id == connection_id,
                s.column_attribute.c.table_id == s.catalog_table.c.id,
            )
        )
    return statement


def _sql_attributes_of_tagged_columns(
    tag_id: str, connection_id: str | None = None
) -> Select:
    """Ids of unprocessed SqlAttributes whose SQL reads a column with *tag_id*.

    Reached through the statement the attribute is computed by:
    ``sql_attribute__sql`` -> ``sql_query__column`` -> the column. Term-linked
    and unprocessed only, for the reasons on
    :func:`_column_attributes_of_tagged_columns`.
    """
    statement = (
        select(s.sql_attribute__sql.c.attribute_id.label("id"))
        .select_from(
            s.tag_target.join(
                s.sql_query__column,
                s.sql_query__column.c.column_id == s.tag_target.c.column_id,
            )
            .join(
                s.sql_attribute__sql,
                s.sql_attribute__sql.c.sql_query_id
                == s.sql_query__column.c.sql_query_id,
            )
            .join(
                s.sql_attribute__term,
                s.sql_attribute__term.c.attribute_id
                == s.sql_attribute__sql.c.attribute_id,
            )
            .join(
                s.sql_attribute,
                s.sql_attribute.c.id == s.sql_attribute__sql.c.attribute_id,
            )
        )
        .where(
            s.tag_target.c.tag_id == tag_id,
            s.sql_attribute.c.pii_processed.is_(False),
        )
        .distinct()
    )
    if connection_id is not None:
        statement = statement.where(
            definition_source_scope(
                s.sql_attribute, s.sql_attribute__sql, "attribute_id", connection_id
            )
        )
    return statement


def tag_attributes_of_tagged_columns(
    tag_id: str, *, connection_id: str | None = None
) -> tuple[int, int]:
    """Copy *tag_id* from tagged columns onto the attributes built from them.

    Returns ``(column_attributes, sql_attributes)`` newly labelled.

    Each attribute is handled **once**. A candidate is an attribute with
    ``pii_processed`` false that is built from a column carrying *tag_id*; it is
    labelled (unless it already is) and then marked processed. It is never a
    candidate again, so a person who removes the label has overridden it for
    good -- the same rule ``catalog_column.pii_processed`` gives columns.

    An attribute built from no tagged column is left unprocessed, so it is still
    picked up if one of its columns is tagged later.

    The source of truth is the label on the column itself, not the detector's
    verdict. A column a person tagged by hand propagates exactly like one the
    detector tagged, and this does not depend on the column's own
    ``pii_processed`` -- which matters because attributes are written by
    semantic compilation, *after* the ingest that classified the column.

    Additive on ingest/compile. Taking the tag off a column is the other
    exception -- see :func:`untag_attributes_of_column` -- as is a rewritten
    SqlAttribute expression that no longer reads a tagged column
    (:func:`realign_sql_attribute_pii`).

    Labels carry no ``tagged_by`` and no ``rule_id``, which the UI reads as
    "Auto Generated". A label already on the attribute keeps its source.

    Runtime passes connection_id: catalog ancestry and every SQL dependency
    must belong to it. Omitting it retains the explicit administrative sweep.
    """
    applied: list[int] = []
    if connection_id is not None and not connection_id.strip():
        raise ValueError("connection_id must not be blank")
    stored = s.tag_target.alias("stored")
    with write_transaction():
        for column, attribute, candidates in (
            (
                s.tag_target.c.column_attribute_id,
                s.column_attribute,
                _column_attributes_of_tagged_columns,
            ),
            (
                s.tag_target.c.sql_attribute_id,
                s.sql_attribute,
                _sql_attributes_of_tagged_columns,
            ),
        ):
            found = candidates(tag_id, connection_id).subquery()
            rows = store().query_write(
                insert(s.tag_target)
                .from_select(
                    ["tag_id", column.name],
                    select(literal(tag_id, Text), found.c.id).where(
                        not_(
                            exists(
                                select(literal(1)).where(
                                    stored.c.tag_id == tag_id,
                                    stored.c[column.name] == found.c.id,
                                )
                            )
                        )
                    ),
                )
                .on_conflict_do_nothing()
                .returning(s.tag_target.c.id)
            )
            applied.append(len(rows))
            # The candidate set again rather than ids read back into Python: a
            # first pass over a large catalog can have more candidates than
            # psycopg's 65535 bind parameters, and an `IN` of that many
            # literals would raise -- silently, since the caller swallows a
            # propagation failure, and permanently, since nothing would then be
            # marked processed to shrink the next pass. The statement's
            # snapshot still sees every candidate unprocessed.
            store().query_write(
                attribute.update()
                .where(attribute.c.id.in_(select(found.c.id)))
                .values(pii_processed=True)
            )
    return applied[0], applied[1]


def _sql_attribute_reads_tagged_column(attr_id: str, tag_id: str) -> bool:
    """True when any SQL for *attr_id* still reads a column carrying *tag_id*."""
    return bool(
        store().query_read(
            select(s.sql_attribute__sql.c.attribute_id)
            .select_from(
                s.tag_target.join(
                    s.sql_query__column,
                    s.sql_query__column.c.column_id == s.tag_target.c.column_id,
                ).join(
                    s.sql_attribute__sql,
                    s.sql_attribute__sql.c.sql_query_id
                    == s.sql_query__column.c.sql_query_id,
                )
            )
            .where(
                s.tag_target.c.tag_id == tag_id,
                s.sql_attribute__sql.c.attribute_id == attr_id,
            )
            .limit(1)
        )
    )


def realign_sql_attribute_pii(attr_id: str, tag_id: str | None) -> bool:
    """Drop *tag_id* after an expression rewrite if the SQL no longer reads it.

    Returns True when the SQL does not read a column carrying *tag_id*: the
    label is taken off (if present) and ``pii_processed`` is cleared so a
    later tag on one of the new columns can still propagate. Returns False
    when the SQL still reads a tagged column: the label and the processed
    flag stay. *tag_id* None means there is no such tag, so no column can
    carry it.
    """
    if tag_id is not None and _sql_attribute_reads_tagged_column(attr_id, tag_id):
        return False
    with write_transaction():
        if tag_id is not None:
            detach_tag(tag_id=tag_id, kind=TARGET_SQL_ATTRIBUTE, item_id=attr_id)
        store().query_write(
            s.sql_attribute.update()
            .where(s.sql_attribute.c.id == attr_id)
            .values(pii_processed=False)
        )
    return True


def untag_attributes_of_column(column_id: str, tag_id: str) -> None:
    """Drop *tag_id* from attributes of *column_id* after the column lost it.

    Call this *after* the column's own label is gone.

    Every ColumnAttribute this column *is* (``column__has_attribute``, not a
    semantic FK) loses the label and ``pii_processed`` is cleared so a later
    tag on the column can still propagate. SQL attributes whose query reads
    this column are realigned: the label stays if they still read another
    tagged column.

    One transaction: a later SQL realign must not leave column attributes
    already untagged if it fails. Nested :func:`write_transaction` calls
    reuse this one.
    """
    with write_transaction():
        column_attr_ids = [
            row["attribute_id"]
            for row in store().query_read(
                select(s.column__has_attribute.c.attribute_id).where(
                    s.column__has_attribute.c.column_id == column_id
                )
            )
        ]
        sql_attr_ids = [
            row["attribute_id"]
            for row in store().query_read(
                select(s.sql_attribute__sql.c.attribute_id)
                .select_from(
                    s.sql_attribute__sql.join(
                        s.sql_query__column,
                        s.sql_query__column.c.sql_query_id
                        == s.sql_attribute__sql.c.sql_query_id,
                    )
                )
                .where(s.sql_query__column.c.column_id == column_id)
                .distinct()
            )
        ]
        if column_attr_ids:
            for attr_id in column_attr_ids:
                detach_tag(
                    tag_id=tag_id,
                    kind=TARGET_COLUMN_ATTRIBUTE,
                    item_id=attr_id,
                )
            store().query_write(
                s.column_attribute.update()
                .where(s.column_attribute.c.id.in_(column_attr_ids))
                .values(pii_processed=False)
            )
        for attr_id in sql_attr_ids:
            realign_sql_attribute_pii(attr_id, tag_id)


def untag_attributes_of_columns_being_deleted(column_ids: list[str]) -> None:
    """Drop PII from attributes before *column_ids* leave the catalog.

    Ingest deletes dropped columns (and whole tables/schemas) without going
    through the tags API. Postgres then cascades the column's own label and
    the SQL / HAS_ATTRIBUTE edges, which would leave PII on the attributes.
    Take PII off the columns first so SQL realign does not still see them as
    tagged, then reuse :func:`untag_attributes_of_column`. One transaction
    covers every detach and every untag; nested scopes reuse it.
    """
    ids = list(dict.fromkeys(column_id for column_id in column_ids if column_id))
    if not ids:
        return
    pii_tag = get_tag_by_name(PII_TAG_NAME)
    if pii_tag is None:
        return
    pii_id = str(pii_tag["id"])
    with write_transaction():
        for column_id in ids:
            detach_tag(tag_id=pii_id, kind=TARGET_COLUMN, item_id=column_id)
        for column_id in ids:
            untag_attributes_of_column(column_id, pii_id)
