# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compilation reads project shared vocabulary onto one catalog root."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert

from auto_ontology.dal import datasources as d, schema as s, terms as t
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database,  # noqa: F401
)


@pytest.fixture
def sources(identity_database):  # noqa: F811
    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    store().query_write(
        insert(s.term).values(id="shared", name="Order", source="semantic")
    )
    for cid in ("a", "b"):
        for table, values in (
            (s.catalog_database, dict(id=cid, name="sales", display_label=cid)),
            (s.catalog_schema, dict(id=f"s-{cid}", name="public", database_id=cid)),
            (
                s.catalog_table,
                dict(
                    id=f"t-{cid}",
                    name="orders",
                    schema_id=f"s-{cid}",
                    description=f"marker-{cid}",
                ),
            ),
            (
                s.catalog_column,
                dict(
                    id=f"c-{cid}",
                    name=f"marker_{cid}",
                    table_id=f"t-{cid}",
                    sample_values=[f"sample-{cid}"],
                ),
            ),
            (
                s.sql_query,
                dict(
                    id=f"q-{cid}",
                    sql_full_query="select * from public.orders",
                    connection_id=cid,
                ),
            ),
            (s.sql_query__table, dict(sql_query_id=f"q-{cid}", table_id=f"t-{cid}")),
            (s.table__term, dict(table_id=f"t-{cid}", term_id="shared")),
        ):
            store().query_write(insert(table).values(**values))
    store().query_write(
        insert(s.sql_query).values(id="legacy", sql_full_query="legacy")
    )
    store().query_write(
        insert(s.sql_query__table).values(sql_query_id="legacy", table_id="t-a")
    )
    yield
    dispose_engine()


def test_compilation_reads_never_mix_same_named_sources(sources):
    assert [r["id"] for r in d.fetch_sorted_tables(connection_id="a")] == ["t-a"]
    assert t.fetch_table_schema_map(connection_id="a") == {"orders": "public"}
    rows = t.fetch_terms_with_sqls(connection_id="a")
    assert [r["sql_id"] for r in rows[0]["sqls"]] == ["q-a"]
    assert {r["table_id"] for r in t.fetch_term_table_pairs(connection_id="a")} == {
        "t-a"
    }
    ctx = d.fetch_table_context("t-a", connection_id="a")
    assert ctx["connection_id"] == "a"
    assert [c["name"] for c in ctx["columns"]] == ["marker_a"]
    with pytest.raises(ValueError):
        d.fetch_table_context("t-b", connection_id="a")
    store().query_write(s.table__term.delete())
    tables = d.fetch_all_tables_without_term(connection_id="a")
    assert [(r["id"], r["connection_id"]) for r in tables] == [("t-a", "a")]


def test_foreign_query_link_excludes_entire_statement(sources):
    store().query_write(
        insert(s.sql_query__table).values(sql_query_id="q-a", table_id="t-b")
    )
    assert t.fetch_terms_with_sqls(connection_id="a") == []


def test_bridge_candidates_and_targets_are_source_scoped(sources):
    for cid in ("a", "b"):
        store().query_write(
            insert(s.catalog_table).values(
                id=f"bridge-{cid}", name="order_links", schema_id=f"s-{cid}"
            )
        )
        for index in (1, 2):
            column = f"link-{cid}-{index}"
            store().query_write(
                insert(s.catalog_column).values(
                    id=column, name=f"order_{index}", table_id=f"bridge-{cid}"
                )
            )
            store().query_write(
                insert(s.column__foreign_key).values(
                    source_column_id=column, target_column_id=f"c-{cid}"
                )
            )
    rows = d.fetch_bridge_table_candidates("a")
    assert [row["table_id"] for row in rows] == ["bridge-a"]
    assert {pair["target_table_id"] for pair in rows[0]["fk_pairs"]} == {"t-a"}
    store().query_write(
        s.column__foreign_key.update()
        .where(s.column__foreign_key.c.source_column_id == "link-a-1")
        .values(target_column_id="c-b")
    )
    assert d.fetch_bridge_table_candidates("a") == []


def test_term_pairs_exclude_foreign_attribute_projection(sources):
    store().query_write(
        insert(s.term).values(id="b-only", name="B concept", source="semantic")
    )
    store().query_write(
        insert(s.column_attribute).values(
            id="attr-b",
            name="B id",
            source="semantic",
            table_id="t-b",
            source_column="marker_b",
            term_name="B concept",
        )
    )
    store().query_write(
        insert(s.column_attribute__term).values(attribute_id="attr-b", term_id="b-only")
    )
    store().query_write(
        insert(s.column__has_attribute).values(attribute_id="attr-b", column_id="c-b")
    )
    store().query_write(
        insert(s.column__semantic_fk).values(attribute_id="attr-b", column_id="c-a")
    )
    assert {r["term_id"] for r in t.fetch_term_table_pairs(connection_id="a")} == {
        "shared"
    }
