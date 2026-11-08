# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FK ownership is catalog ancestry, never a physical database name."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert, select

from auto_ontology.dal import attributes as a, schema as s
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database,  # noqa: F401
)


@pytest.fixture
def sources(identity_database):  # noqa: F811
    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    for cid in ("a", "b"):
        for table, values in (
            (s.catalog_database, dict(id=cid, name="sales", display_label=cid)),
            (s.catalog_schema, dict(id=f"s-{cid}", name="public", database_id=cid)),
            (
                s.catalog_table,
                dict(id=f"orders-{cid}", name="orders", schema_id=f"s-{cid}"),
            ),
            (
                s.catalog_table,
                dict(id=f"customers-{cid}", name="customers", schema_id=f"s-{cid}"),
            ),
            (
                s.catalog_column,
                dict(
                    id=f"customer-{cid}",
                    name="customer_id",
                    table_id=f"orders-{cid}",
                    sample_values=[f"marker-{cid}"],
                ),
            ),
            (
                s.catalog_column,
                dict(id=f"free-{cid}", name="open_id", table_id=f"orders-{cid}"),
            ),
            (
                s.catalog_column,
                dict(id=f"bad-{cid}", name="foreign_id", table_id=f"orders-{cid}"),
            ),
            (
                s.catalog_column,
                dict(id=f"target-{cid}", name="id", table_id=f"customers-{cid}"),
            ),
            (
                s.column_attribute,
                dict(
                    id=f"attr-{cid}",
                    name=f"marker-{cid}",
                    table_id=f"customers-{cid}",
                    source_column="id",
                    term_name="Customer",
                ),
            ),
            (
                s.column__has_attribute,
                dict(column_id=f"target-{cid}", attribute_id=f"attr-{cid}"),
            ),
            (
                s.column__foreign_key,
                dict(
                    source_column_id=f"customer-{cid}", target_column_id=f"target-{cid}"
                ),
            ),
        ):
            store().query_write(insert(table).values(**values))
    store().query_write(
        insert(s.column__foreign_key).values(
            source_column_id="bad-a", target_column_id="target-b"
        )
    )
    yield
    dispose_engine()


def edges():
    return store().query_read(
        select(s.column__semantic_fk).order_by(s.column__semantic_fk.c.column_id)
    )


def test_foreign_merge_with_same_physical_name_is_rejected_without_write(sources):
    assert a.merge_semantic_fk("customer-b", "attr-b")
    before = edges()
    assert a.merge_semantic_fk("customer-a", "attr-b") is False
    assert edges() == before


def test_local_merge_is_idempotent(sources):
    assert a.merge_semantic_fk("customer-a", "attr-a") is True
    assert a.merge_semantic_fk("customer-a", "attr-a") is True
    assert [(row["column_id"], row["attribute_id"]) for row in edges()] == [
        ("customer-a", "attr-a")
    ]


def test_guard_uses_canonical_owner_not_reverse_fk_fallback(sources):
    store().query_write(
        s.column__has_attribute.delete().where(
            s.column__has_attribute.c.attribute_id == "attr-b"
        )
    )
    store().query_write(
        insert(s.column__semantic_fk).values(column_id="free-a", attribute_id="attr-b")
    )
    before = edges()
    assert a.merge_semantic_fk("customer-a", "attr-b") is False
    assert edges() == before


def test_unlinked_candidates_exclude_foreign_declared_target_as_whole(sources):
    rows = a.find_unlinked_fk_columns(connection_id="a")
    assert {row["id"] for row in rows} == {"customer-a", "free-a"}
    assert {row["connection_id"] for row in rows} == {"a"}
    assert {row["fk_target_col_id"] for row in rows} == {"target-a", None}
    assert "marker-b" not in repr(rows)


def test_one_local_target_does_not_hide_another_foreign_declared_target(sources):
    store().query_write(
        insert(s.column__foreign_key).values(
            source_column_id="bad-a", target_column_id="target-a"
        )
    )
    assert "bad-a" not in {
        row["id"] for row in a.find_unlinked_fk_columns(connection_id="a")
    }


def test_join_groups_exclude_corrupted_foreign_and_noncanonical_links(sources):
    for column, attribute in [
        ("customer-a", "attr-a"),
        ("free-a", "attr-b"),
        ("customer-b", "attr-b"),
    ]:
        store().query_write(
            insert(s.column__semantic_fk).values(
                column_id=column, attribute_id=attribute
            )
        )
    store().query_write(
        insert(s.column__has_attribute).values(
            column_id="free-a", attribute_id="attr-b"
        )
    )
    rows = a.fetch_semantic_fk_join_groups(connection_id="a")
    assert [(row["source_column_id"], row["target_column_id"]) for row in rows] == [
        ("customer-a", "target-a")
    ]
    assert {row["connection_id"] for row in rows} == {"a"}


def test_scoped_attribute_context_never_decorates_with_foreign_metadata(sources):
    store().query_write(
        insert(s.column__semantic_fk).values(column_id="free-a", attribute_id="attr-b")
    )
    store().query_write(
        insert(s.column__has_attribute).values(
            column_id="free-a", attribute_id="attr-b"
        )
    )
    contexts = a.fetch_attr_column_contexts(["attr-a", "attr-b"], connection_id="a")
    assert set(contexts) == {"attr-a"}
    assert contexts["attr-a"]["connection_id"] == "a"
    assert contexts["attr-a"]["col_id"] == "target-a"
    assert "marker-b" not in repr(contexts)


def test_unresolved_canonical_owner_cannot_use_reverse_or_owned_fallback(sources):
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(table_id="")
    )
    assert a.merge_semantic_fk("customer-a", "attr-a") is False
    assert a.fetch_attr_column_contexts(["attr-a"], connection_id="a") == {}
    assert edges() == []


def test_legacy_name_context_filter_does_not_leak_other_catalog_fields(sources):
    assert a.fetch_attr_column_contexts(["attr-b"], database_name="other") == {}
