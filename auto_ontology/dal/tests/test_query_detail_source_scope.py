# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query detail traversal on disposable, source-owned metadata (no source SQL)."""

from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import Table, insert

from auto_ontology.dal import exploration as e, schema as s
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_public_semantic_source_scope import (
    identity_database as identity_database,
    semantic_sources as semantic_sources,
)


@pytest.mark.usefixtures("semantic_sources")
@pytest.mark.parametrize(
    ("reader", "node_id", "key", "expected_id"),
    [
        (e.fetch_column_exploration_details, "column-a", "column_attribute", "attr-a"),
        (e.fetch_column_attribute_exploration_details, "attr-a", "term", "shared"),
        (e.fetch_sql_attribute_exploration_details, "sql-attr-a", "sql", "sql-a"),
    ],
)
def test_detail_callers_accept_source_and_keep_local_metadata(
    reader: Callable[..., dict[str, Any]], node_id: str, key: str, expected_id: str
) -> None:
    assert reader(node_id, connection_id="a")[key]["id"] == expected_id


@pytest.mark.usefixtures("semantic_sources")
@pytest.mark.parametrize("link", [s.column__has_attribute, s.column__semantic_fk])
def test_local_column_does_not_hydrate_foreign_attribute(link: Table) -> None:
    store().query_write(
        s.column__has_attribute.delete().where(
            s.column__has_attribute.c.column_id == "column-a"
        )
    )
    store().query_write(
        insert(link).values(column_id="column-a", attribute_id="attr-b")
    )
    result = e.fetch_column_exploration_details("column-a", connection_id="a")
    assert result["column_attribute"] is None
    assert "marker-b" not in repr(result)
    assert (
        e.fetch_column_exploration_details("column-a")["column_attribute"]["id"]
        == "attr-b"
    )


@pytest.mark.usefixtures("semantic_sources")
def test_column_scopes_both_fk_ends_and_sql_ownership() -> None:
    store().query_write(
        insert(s.catalog_column).values(
            id="local-fk", name="parent", table_id="table-a"
        )
    )
    for source, target in [
        ("local-fk", "column-a"),
        ("column-b", "column-a"),
        ("column-a", "column-b"),
    ]:
        store().query_write(
            insert(s.column__foreign_key).values(
                source_column_id=source, target_column_id=target
            )
        )
    for query in ("sql-a", "sql-b"):
        store().query_write(
            insert(s.sql_query__column).values(sql_query_id=query, column_id="column-a")
        )
    result = e.fetch_column_exploration_details("column-a", connection_id="a")
    assert result["column_attribute"]["id"] == "attr-a"
    assert result["foreign_key_column"] is None
    assert [row["id"] for row in result["referencing_columns"]] == ["local-fk"]
    assert [row["id"] for row in result["sql_queries"]] == ["sql-a"]
    assert (
        e.fetch_column_exploration_details("local-fk", connection_id="a")[
            "foreign_key_column"
        ]["id"]
        == "column-a"
    )
    assert e.fetch_column_exploration_details("column-b", connection_id="a") == {
        "column_attribute": None,
        "foreign_key_column": None,
        "referencing_columns": [],
        "sql_queries": [],
    }
    assert (
        e.fetch_column_exploration_details("column-a")["foreign_key_column"]["id"]
        == "column-b"
    )


@pytest.mark.usefixtures("semantic_sources")
def test_attribute_count_and_page_only_include_valid_local_paths() -> None:
    store().query_write(
        insert(s.catalog_table).values(id="other-a", name="other", schema_id="schema-a")
    )
    store().query_write(
        insert(s.catalog_column).values(
            id="local-fk", name="parent", table_id="other-a"
        )
    )
    for column in ("local-fk", "column-b"):
        store().query_write(
            insert(s.column__semantic_fk).values(
                column_id=column, attribute_id="attr-a"
            )
        )
    # A local HAS_ATTRIBUTE edge to the wrong canonical column must not be counted.
    store().query_write(
        insert(s.column__has_attribute).values(
            column_id="local-fk", attribute_id="attr-a"
        )
    )
    page = e.fetch_column_attribute_exploration_details(
        "attr-a", connection_id="a", skip=1, limit=1
    )
    assert page["term"]["id"] == "shared"
    assert page["columns_total"] == 2
    assert [row["id"] for row in page["columns"]] == ["local-fk"]
    assert page["columns"][0]["relationship_types"] == ["SEMANTIC_FK"]
    assert "marker-b" not in repr(page)
    assert (
        e.fetch_column_attribute_exploration_details(
            "attr-a", connection_id="a", skip=2, limit=1
        )["columns"]
        == []
    )
    assert (
        e.fetch_column_exploration_details("local-fk", connection_id="a")[
            "column_attribute"
        ]["id"]
        == "attr-a"
    )
    assert e.fetch_column_attribute_exploration_details(
        "attr-b", connection_id="a"
    ) == {"term": None, "columns": [], "columns_total": 0}


@pytest.mark.usefixtures("semantic_sources")
@pytest.mark.parametrize(
    "corruption", ["foreign-owner", "wrong-column", "missing-canonical"]
)
def test_attribute_canonical_ownership_precedes_description_and_term(
    corruption: str,
) -> None:
    if corruption == "foreign-owner":
        store().query_write(
            s.column_attribute.update()
            .where(s.column_attribute.c.id == "attr-a")
            .values(table_id="table-b", description="marker-b")
        )
    elif corruption == "wrong-column":
        store().query_write(
            s.column_attribute.update()
            .where(s.column_attribute.c.id == "attr-a")
            .values(source_column="not-value", description="marker-b")
        )
    else:
        store().query_write(
            s.column__has_attribute.delete().where(
                s.column__has_attribute.c.attribute_id == "attr-a"
            )
        )
        store().query_write(
            insert(s.column__semantic_fk).values(
                column_id="column-a", attribute_id="attr-a"
            )
        )
    assert (
        e.fetch_column_exploration_details("column-a", connection_id="a")[
            "column_attribute"
        ]
        is None
    )
    assert e.fetch_column_attribute_exploration_details(
        "attr-a", connection_id="a"
    ) == {"term": None, "columns": [], "columns_total": 0}


@pytest.mark.usefixtures("semantic_sources")
@pytest.mark.parametrize(
    "corruption",
    [
        "table",
        "column",
        "query-owner",
        "query-null",
        "definition-owner",
        "definition-null",
        "mixed-definition",
    ],
)
def test_sql_definition_rejects_every_foreign_dependency(corruption: str) -> None:
    if corruption in {"table", "column"}:
        link = s.sql_query__table if corruption == "table" else s.sql_query__column
        key = "table_id" if corruption == "table" else "column_id"
        store().query_write(
            insert(link).values(sql_query_id="sql-a", **{key: f"{corruption}-b"})
        )
    elif corruption.startswith("query-"):
        store().query_write(
            s.sql_query.update()
            .where(s.sql_query.c.id == "sql-a")
            .values(connection_id="b" if corruption.endswith("owner") else None)
        )
    elif corruption.startswith("definition-"):
        store().query_write(
            s.sql_attribute.update()
            .where(s.sql_attribute.c.id == "sql-attr-a")
            .values(connection_id="b" if corruption.endswith("owner") else None)
        )
    else:
        store().query_write(
            insert(s.sql_attribute__sql).values(
                attribute_id="sql-attr-a", sql_query_id="sql-b"
            )
        )
    assert e.fetch_sql_attribute_exploration_details(
        "sql-attr-a", connection_id="a"
    ) == {"sql": None, "term": None}
    assert e.fetch_sql_attribute_exploration_details("sql-attr-a")["sql"] is not None
    assert e.fetch_sql_attribute_exploration_details(
        "sql-attr-b", connection_id="a"
    ) == {"sql": None, "term": None}


@pytest.mark.usefixtures("semantic_sources")
@pytest.mark.parametrize("dependency", ["table", "column", "null-owner"])
def test_column_sql_rejects_foreign_dependencies_and_unresolved_owner(
    dependency: str,
) -> None:
    store().query_write(
        insert(s.sql_query__column).values(sql_query_id="sql-a", column_id="column-a")
    )
    if dependency == "null-owner":
        store().query_write(
            s.sql_query.update()
            .where(s.sql_query.c.id == "sql-a")
            .values(connection_id=None)
        )
    else:
        link = s.sql_query__table if dependency == "table" else s.sql_query__column
        key = "table_id" if dependency == "table" else "column_id"
        store().query_write(
            insert(link).values(sql_query_id="sql-a", **{key: f"{dependency}-b"})
        )
    result = e.fetch_column_exploration_details("column-a", connection_id="a")
    assert result["column_attribute"]["id"] == "attr-a"
    assert result["sql_queries"] == []
