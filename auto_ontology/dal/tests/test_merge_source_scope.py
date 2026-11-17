# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DB-free ownership boundaries for the upstream retrieval/compile additions."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from sqlalchemy.dialects import postgresql

from auto_ontology.dal import attributes, datasources, term_families, terms
from auto_ontology.dal import schema as s
from auto_ontology.dal.source_scope import query_source_scope
from auto_ontology.dal import search


def sql(statement):
    return str(
        statement.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


@pytest.mark.parametrize(
    "function",
    [
        term_families.fetch_family_candidates,
        term_families.fetch_family_edges,
        term_families.fetch_table_families,
        attributes.fetch_value_key_candidates,
        attributes.fetch_attributes_to_describe,
        attributes.fetch_semantic_fk_ends,
        datasources.fetch_table_column_names,
        datasources.fetch_schema_ids_for_database,
    ],
)
def test_new_catalog_reads_select_identity(monkeypatch, function):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(function.__module__ + ".store", lambda: database)
    function("sales", connection_id="source-a")
    statement = sql(database.query_read.call_args.args[0])
    assert "database_id = 'source-a'" in statement


def test_family_replace_rejects_foreign_schema_before_delete(monkeypatch):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(term_families, "store", lambda: database)
    monkeypatch.setattr(term_families, "write_transaction", nullcontext)
    with pytest.raises(ValueError):
        term_families.replace_term_families(
            "sales", [("foreign", "p", {"term": "base"})], connection_id="source-a"
        )
    database.query_write.assert_not_called()


def test_family_replace_rejects_foreign_term_before_delete(monkeypatch):
    database = Mock()
    database.query_read.side_effect = [[{"id": "schema-a"}], []]
    monkeypatch.setattr(term_families, "store", lambda: database)
    monkeypatch.setattr(term_families, "write_transaction", nullcontext)
    with pytest.raises(ValueError, match="members"):
        term_families.replace_term_families(
            "sales",
            [("schema-a", "p", {"foreign-term": "base"})],
            connection_id="source-a",
        )
    database.query_write.assert_not_called()


def test_semantic_coverage_scopes_both_ids(monkeypatch):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(terms, "store", lambda: database)
    assert terms.fetch_semantic_coverage(
        ["foreign-column"], ["foreign-table"], connection_id="source-a"
    ) == (set(), set())
    assert len(database.query_read.call_args_list) == 2
    for call in database.query_read.call_args_list:
        assert "database_id = 'source-a'" in sql(call.args[0])


def test_description_write_excludes_foreign_attributes(monkeypatch):
    database = Mock()
    database.query_write.return_value = []
    monkeypatch.setattr(attributes, "store", lambda: database)
    assert (
        attributes.store_attribute_descriptions(
            {"foreign": "text"}, connection_id="source-a"
        )
        == 0
    )
    statement = sql(database.query_write.call_args.args[0])
    assert "database_id = 'source-a'" in statement
    assert "column_attribute.table_id IN" in statement


def test_family_replace_delete_is_scoped(monkeypatch):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(term_families, "store", lambda: database)
    monkeypatch.setattr(term_families, "write_transaction", nullcontext)
    assert (
        term_families.replace_term_families("sales", [], connection_id="source-a") == 0
    )
    assert "database_id = 'source-a'" in sql(database.query_write.call_args.args[0])


def test_family_edges_reject_unknown_and_mixed_sql_ownership(monkeypatch):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(term_families, "store", lambda: database)
    term_families.fetch_family_edges("sales", connection_id="source-a")
    statement = sql(database.query_read.call_args.args[0])
    assert "sql_query.connection_id = 'source-a'" in statement
    assert "database_id IS DISTINCT FROM 'source-a'" in statement
    assert "sql_query__column" in statement
    assert "sql_query__table" in statement


def test_query_scope_fails_closed_on_null_dependencies():
    statement = sql(s.sql_query.select().where(query_source_scope("source-a")))
    assert statement.count("database_id IS DISTINCT FROM 'source-a'") == 2
    assert "LEFT OUTER JOIN catalog_schema" in statement


def test_table_context_rejects_foreign_id_before_reading_columns(monkeypatch):
    database = Mock()
    database.query_read.return_value = [{"database_id": "source-b"}]
    monkeypatch.setattr(datasources, "store", lambda: database)
    with pytest.raises(ValueError, match="connection_id"):
        datasources.fetch_table_context("foreign", connection_id="source-a")
    database.query_read.assert_called_once()


def test_name_search_scopes_before_limit(monkeypatch):
    database = Mock()
    database.query_read.return_value = []
    monkeypatch.setattr(search, "store", lambda: database)
    assert (
        search.fetch_global_search(
            ["orders"],
            {"Table", "Term", "SqlAttribute"},
            include_description=True,
            connection_id="source-a",
        )
        == []
    )
    statement = sql(database.query_read.call_args.args[0])
    assert "database_id = 'source-a'" in statement
    assert "connection_id = 'source-a'" in statement
