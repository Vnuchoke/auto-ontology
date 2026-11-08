# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable SQL belongs to exactly one registered catalog root."""

from datetime import datetime

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert, select

from auto_ontology.catalog.constants import Labels, Props
from auto_ontology.catalog.model.node import CatalogNode
from auto_ontology.catalog.model.query import Query
from auto_ontology.catalog.store.queries import (
    add_query,
    get_sql_by_full_query,
    load_sqls_to_tables,
    update_counters_and_timestamps_for_query_and_affected_data,
)
from auto_ontology.catalog.store.rows import upsert_row
from auto_ontology.catalog.store.registry import LABEL_SQL_ATTRIBUTE
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database,  # noqa: F401
)


@pytest.fixture
def sources(identity_database):  # noqa: F811
    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    for owner in ("a", "b"):
        store().query_write(
            insert(s.catalog_database).values(
                id=owner, name="sales", display_label=owner
            )
        )
        store().query_write(
            insert(s.catalog_schema).values(
                id=f"s-{owner}", name="public", database_id=owner
            )
        )
        store().query_write(
            insert(s.catalog_table).values(
                id=f"t-{owner}", name="orders", schema_id=f"s-{owner}"
            )
        )
        store().query_write(
            insert(s.catalog_column).values(
                id=f"c-{owner}", name="id", table_id=f"t-{owner}"
            )
        )
    yield
    dispose_engine()


def query(owner, text="select id from orders"):
    return Query({}, f"q-{owner}", text, datetime.now(), 1, connection_id=owner)


def edges(q, owner):
    return [
        (
            q.sql_node,
            CatalogNode("orders", Labels.TABLE, {}, existing_id=f"t-{owner}"),
            {Props.SQL_ID: q.id},
        ),
        (
            q.sql_node,
            CatalogNode("id", Labels.COLUMN, {}, existing_id=f"c-{owner}"),
            {Props.SQL_ID: q.id},
        ),
    ]


def test_identical_text_counters_and_legacy_are_source_scoped(sources):
    for owner in ("a", "b"):
        add_query(edges(query(owner), owner), connection_id=owner)
    store().query_write(
        insert(s.sql_query).values(id="legacy", sql_full_query="legacy")
    )
    assert get_sql_by_full_query("select id from orders", connection_id="a") == "q-a"
    assert get_sql_by_full_query("legacy", connection_id="a") is None
    assert load_sqls_to_tables(connection_id="a").iloc[0]["cols"] == ["c-a"]
    update_counters_and_timestamps_for_query_and_affected_data(
        "q-b", query("a").sql_node, connection_id="a"
    )
    assert (
        store().query_read(
            select(s.sql_query.c.total_counter).where(s.sql_query.c.id == "q-b")
        )[0]["total_counter"]
        == 1
    )
    update_counters_and_timestamps_for_query_and_affected_data(
        "q-a", query("a").sql_node, connection_id="a"
    )
    assert (
        store().query_read(
            select(s.sql_query.c.total_counter).where(s.sql_query.c.id == "q-a")
        )[0]["total_counter"]
        == 2
    )


@pytest.mark.parametrize("target", ["b", "missing"])
def test_all_dependencies_checked_before_any_write(sources, target):
    q = query("a")
    before = store().query_read(select(s.sql_query))
    with pytest.raises(ValueError):
        add_query(edges(q, "a") + edges(q, target), connection_id="a")
    assert store().query_read(select(s.sql_query)) == before


@pytest.mark.parametrize(
    "label, table",
    [
        (Labels.SQL, s.sql_query),
        (LABEL_SQL_ATTRIBUTE, s.sql_attribute),
        (Labels.CUSTOM_ANALYSIS, s.custom_analysis),
    ],
)
@pytest.mark.parametrize("owner", ["b", None])
def test_existing_definition_id_cannot_be_reassigned(sources, label, table, owner):
    values = {"id": "foreign", "name": "definition", "connection_id": owner}
    if table is s.sql_query:
        values.pop("name")
        values["sql_full_query"] = "select 42"
    store().query_write(insert(table).values(**values))
    before = store().query_read(select(table))
    with pytest.raises(ValueError):
        upsert_row(
            label,
            {"id": "foreign", "connection_id": "a"},
            {**values, "connection_id": "a"},
        )
    assert store().query_read(select(table)) == before


def test_new_generated_query_id_uses_stored_id_in_dependencies(sources):
    original = query("a")
    add_query(edges(original, "a"), connection_id="a")
    duplicate = Query(
        {}, "generated", "select id from orders", datetime.now(), 1, connection_id="a"
    )
    dependencies = edges(duplicate, "a")
    dependencies.append(
        (
            dependencies[1][1],
            dependencies[1][1],
            {Props.JOIN: True, "refs": ["generated|id=id"]},
        )
    )
    add_query(dependencies, connection_id="a")
    assert store().query_read(select(s.column__join.c.refs))[0]["refs"] == ["q-a|id=id"]
    assert duplicate.sql_node.get_id() == original.sql_node.get_id()


@pytest.mark.parametrize("kind", [Props.JOIN, Props.UNION])
def test_mixed_source_column_links_are_atomic(sources, kind):
    q = query("a")
    dependencies = edges(q, "a")
    dependencies.append(
        (dependencies[1][1], edges(q, "b")[1][1], {kind: True, "refs": ["q-a|id=id"]})
    )
    with pytest.raises(ValueError):
        add_query(dependencies, connection_id="a")
    assert store().query_read(select(s.sql_query)) == []


def test_history_exact_and_structural_dedup_never_updates_other_sources(sources):
    from auto_ontology.catalog.sql_parse import populate_queries
    from auto_ontology.server.sql_utils import get_schemas

    for owner in ("a", "b"):
        store().query_write(
            s.catalog_column.update()
            .where(s.catalog_column.c.id == f"c-{owner}")
            .values(data_type="integer")
        )
        schemas = get_schemas(owner)
        assert schemas["public"].get_db_node().id == owner
        df = pd.DataFrame(
            [
                {
                    "query_text": "select id from orders where id = 1",
                    "end_time": datetime.now(),
                    "count": 1,
                }
            ]
        )
        assert populate_queries(schemas, df, 1, ["postgres"], connection_id=owner) == []
    b_before = store().query_read(
        select(s.sql_query).where(s.sql_query.c.connection_id == "b")
    )
    store().query_write(
        insert(s.sql_query).values(
            id="legacy",
            sql_full_query="select id from orders where id = 2",
            total_counter=7,
        )
    )
    for literal in (1, 2):
        df = pd.DataFrame(
            [
                {
                    "query_text": f"select id from orders where id = {literal}",
                    "end_time": datetime.now(),
                    "count": 2,
                }
            ]
        )
        assert (
            populate_queries(get_schemas("a"), df, 1, ["postgres"], connection_id="a")
            == []
        )
    assert (
        store().query_read(
            select(s.sql_query).where(s.sql_query.c.connection_id == "b")
        )
        == b_before
    )
    assert store().query_read(
        select(s.sql_query.c.total_counter).where(s.sql_query.c.connection_id == "a")
    ) == [{"total_counter": 5}]
    assert store().query_read(
        select(s.sql_query.c.total_counter).where(s.sql_query.c.id == "legacy")
    ) == [{"total_counter": 7}]


@pytest.mark.parametrize(
    "label, table",
    [
        (LABEL_SQL_ATTRIBUTE, s.sql_attribute),
        (Labels.CUSTOM_ANALYSIS, s.custom_analysis),
    ],
)
def test_definition_names_are_source_scoped_and_never_claim_legacy(
    sources, label, table
):
    store().query_write(insert(table).values(id="legacy", name="same"))
    ids = [
        upsert_row(
            label,
            {"connection_id": owner, "name": "same"},
            {"connection_id": owner, "name": "same"},
        )
        for owner in ("a", "b")
    ]
    assert len(set(ids + ["legacy"])) == 3
    assert (
        upsert_row(
            label,
            {"connection_id": "a", "name": "same"},
            {"connection_id": "a", "name": "same"},
        )
        == ids[0]
    )


@pytest.mark.parametrize(
    "label, table",
    [
        (LABEL_SQL_ATTRIBUTE, s.sql_attribute),
        (Labels.CUSTOM_ANALYSIS, s.custom_analysis),
    ],
)
def test_foreign_definition_link_rejects_entire_query(sources, label, table):
    store().query_write(
        insert(table).values(id="foreign", connection_id="b", name="definition")
    )
    q = query("a")
    attr = CatalogNode(
        "definition",
        label,
        {"connection_id": "a"},
        existing_id="foreign",
        match_props={"id": "foreign", "connection_id": "a"},
    )
    with pytest.raises(ValueError):
        add_query(
            edges(q, "a") + [(attr, q.sql_node, {Props.ANALYSIS_ID: "foreign"})],
            connection_id="a",
        )
    assert store().query_read(select(s.sql_query)) == []
    assert store().query_read(select(table.c.connection_id)) == [{"connection_id": "b"}]


@pytest.mark.parametrize("owner", [None, ""])
def test_missing_owner_never_reads_or_updates_legacy(sources, owner):
    store().query_write(
        insert(s.sql_query).values(
            id="legacy", sql_full_query="legacy", total_counter=7
        )
    )
    with pytest.raises(ValueError):
        get_sql_by_full_query("legacy", connection_id=owner)
    with pytest.raises(ValueError):
        load_sqls_to_tables(connection_id=owner)
    with pytest.raises(ValueError):
        update_counters_and_timestamps_for_query_and_affected_data(
            "legacy", query("a").sql_node, connection_id=owner
        )
    assert store().query_read(select(s.sql_query.c.total_counter)) == [
        {"total_counter": 7}
    ]
