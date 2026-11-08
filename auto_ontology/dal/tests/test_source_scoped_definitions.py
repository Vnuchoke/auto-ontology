# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.dal import custom_analyses as ca, schema as s, sql_attributes as sa
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)


@pytest.fixture
def definitions(identity_database: None) -> Iterator[None]:
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        store().query_write(s.term.insert().values(id="term", name="Shared revenue"))
        for source in ("a", "b"):
            store().query_write(
                s.catalog_database.insert().values(
                    id=source, name="sales", display_label=source
                )
            )
            store().query_write(
                s.catalog_schema.insert().values(
                    id=f"s-{source}", database_id=source, name="public"
                )
            )
            store().query_write(
                s.catalog_table.insert().values(
                    id=f"t-{source}", schema_id=f"s-{source}", name="orders"
                )
            )
            store().query_write(
                s.catalog_column.insert().values(
                    id=f"c-{source}",
                    table_id=f"t-{source}",
                    name="amount",
                    data_type="int",
                )
            )
        for source in ("a", "b", None):
            suffix = source or "legacy"
            store().query_write(
                s.sql_query.insert().values(
                    id=f"q-{suffix}",
                    connection_id=source,
                    sql_full_query="SELECT amount FROM orders",
                )
            )
            for table in (s.sql_attribute, s.custom_analysis):
                store().query_write(
                    table.insert().values(
                        id=f"{table.name}-{suffix}",
                        connection_id=source,
                        name="Revenue",
                        description=f"{suffix} only",
                    )
                )
            store().query_write(
                s.sql_attribute__term.insert().values(
                    attribute_id=f"sql_attribute-{suffix}", term_id="term"
                )
            )
            store().query_write(
                s.sql_attribute__sql.insert().values(
                    attribute_id=f"sql_attribute-{suffix}", sql_query_id=f"q-{suffix}"
                )
            )
            store().query_write(
                s.custom_analysis__sql.insert().values(
                    analysis_id=f"custom_analysis-{suffix}", sql_query_id=f"q-{suffix}"
                )
            )
            if source:
                store().query_write(
                    s.sql_query__table.insert().values(
                        sql_query_id=f"q-{source}", table_id=f"t-{source}"
                    )
                )
                store().query_write(
                    s.sql_query__column.insert().values(
                        sql_query_id=f"q-{source}", column_id=f"c-{source}"
                    )
                )
        yield
    finally:
        dispose_engine()


@pytest.mark.usefixtures("definitions")
def test_no_source_never_feeds_global_definitions_to_prompts() -> None:
    assert ca.fetch_custom_analyses() == []
    assert ca.fetch_custom_analyses_with_sql(["custom_analysis-legacy"]) == []
    assert sa.fetch_sql_attributes_with_sql(["sql_attribute-legacy"]) == []


@pytest.mark.usefixtures("definitions")
def test_scoped_lists_counts_and_conflicts_keep_shared_term_global() -> None:
    assert [
        r["id"] for r in sa.fetch_sql_attributes_by_term_id("term", connection_id="a")
    ] == ["sql_attribute-a"]
    assert sa.count_sql_attributes_by_term_id("term", connection_id="a") == 1
    assert sa.fetch_sql_attribute_counts(term_ids=["term"], connection_id="b") == [
        {"term_id": "term", "count": 1}
    ]
    assert [r["id"] for r in ca.list_custom_analyses(connection_id="a")] == [
        "custom_analysis-a"
    ]
    assert ca.count_custom_analyses(connection_id="a") == 1
    assert (
        sa.find_attr_by_name("Revenue", None, connection_id="b")["id"]
        == "sql_attribute-b"
    )
    assert (
        ca.find_analysis_by_name("Revenue", None, connection_id="b")["id"]
        == "custom_analysis-b"
    )
    assert (
        ca.find_analysis_by_sql("SELECT amount FROM orders", None, connection_id="b")[
            "id"
        ]
        == "custom_analysis-b"
    )
    assert (
        len(sa.fetch_sql_attributes_by_term_id("term")) == 3
    )  # admin inspection retains legacy


@pytest.mark.usefixtures("definitions")
@pytest.mark.parametrize("foreign_dependency", ["column", "table", "query"])
def test_foreign_or_unresolved_links_are_not_query_context(
    foreign_dependency: str,
) -> None:
    ids = ["sql_attribute-a", "sql_attribute-b", "sql_attribute-legacy"]
    assert [
        r["id"] for r in sa.fetch_sql_attributes_with_sql(ids, connection_id="a")
    ] == ["sql_attribute-a"]
    assert [
        r["id"] for r in sa.fetch_tables_from_sql_attributes(ids, connection_id="a")
    ] == ["t-a"]
    assert ca.fetch_custom_analyses(connection_id="a") == [
        {"name": "Revenue", "description": "a only\nSQL: SELECT amount FROM orders"}
    ]
    # Deliberately corrupted legacy links must not become a partial prompt context.
    if foreign_dependency == "column":
        store().query_write(
            s.sql_query__column.insert().values(sql_query_id="q-a", column_id="c-b")
        )
    elif foreign_dependency == "table":
        store().query_write(
            s.sql_query__table.insert().values(sql_query_id="q-a", table_id="t-b")
        )
    else:
        store().query_write(
            s.sql_attribute__sql.insert().values(
                attribute_id="sql_attribute-a", sql_query_id="q-b"
            )
        )
        store().query_write(
            s.custom_analysis__sql.insert().values(
                analysis_id="custom_analysis-a", sql_query_id="q-b"
            )
        )
    assert sa.fetch_sql_attributes_with_sql(ids, connection_id="a") == []
    assert sa.fetch_tables_from_sql_attributes(ids, connection_id="a") == []
    assert ca.fetch_custom_analyses(connection_id="a") == []
    assert ca.fetch_database_name_for_analysis("custom_analysis-a") is None
    assert ca.list_custom_analyses(connection_id="b")[0]["id"] == "custom_analysis-b"


@pytest.mark.usefixtures("definitions")
def test_deleting_analysis_does_not_delete_shared_or_neighbor_sql() -> None:
    ca.delete_custom_analysis_node("custom_analysis-a")
    assert store().query_read(s.sql_query.select().where(s.sql_query.c.id == "q-a")), (
        "SQL still belongs to the attribute"
    )
    assert (
        sa.fetch_sql_attributes_with_sql(["sql_attribute-a"], connection_id="a")[0][
            "sql"
        ]
        == "SELECT amount FROM orders"
    )
    assert (
        ca.fetch_custom_analyses_with_sql(["custom_analysis-b"], connection_id="b")[0][
            "sql"
        ]
        == "SELECT amount FROM orders"
    )
