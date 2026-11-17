# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react import run_sql, siblings


def test_same_name_sources_a_b_a_have_scoped_catalogs(monkeypatch):
    reads = []

    def ids(name, *, connection_id):
        reads.append(connection_id)
        return [connection_id]

    def schemas(ids, *, connection_id):
        assert ids == [connection_id]
        return {connection_id: object()}

    monkeypatch.setattr(run_sql, "fetch_schema_ids_for_database", ids)
    monkeypatch.setattr(run_sql, "get_schemas_by_ids", schemas)
    connectors = [SimpleNamespace(connection_id=c, database_name="sales") for c in "ab"]
    for cid in "aba":
        ctx = ToolContext(
            retriever=None,
            connectors=connectors,
            database_name="sales",
            connection_id=cid,
        )
        selected = run_sql.connector_for(ctx)
        assert selected.connection_id == cid
        assert set(run_sql.catalog_for(ctx, selected)._schemas) == {cid}
        run_sql.catalog_for(ctx, selected)
    assert reads == list("aba")


def test_missing_selected_connector_never_falls_back():
    ctx = ToolContext(
        retriever=None,
        connectors=[SimpleNamespace(connection_id="b", database_name="sales")],
        database_name="sales",
        connection_id="a",
    )
    assert run_sql.connector_for(ctx) is None


def test_sibling_catalog_is_source_scoped(monkeypatch):
    fetch = Mock(return_value=[])
    monkeypatch.setattr(siblings, "fetch_table_column_names", fetch)
    ctx = ToolContext(
        retriever=None, connectors=[], database_name="sales", connection_id="a"
    )
    assert siblings._catalog(ctx) == []
    fetch.assert_called_once_with("sales", connection_id="a")


@pytest.mark.parametrize("foreign", [None, "b"])
def test_null_or_foreign_dependency_never_routes(foreign):
    from auto_ontology.retrieval.text_to_sql.connector_routing import (
        resolve_connector_from_tables,
    )

    connector = SimpleNamespace(connection_id="a", database_name="sales")
    with pytest.raises(ValueError):
        resolve_connector_from_tables(
            [{"connection_id": "a"}, {"connection_id": foreign}], [connector]
        )


@pytest.mark.parametrize("label", ["SqlAttribute", "CustomAnalysis"])
def test_forged_or_mixed_definition_is_not_hydrated(monkeypatch, label):
    from auto_ontology.retrieval.data_access import candidates

    fetch = Mock(return_value=[])
    monkeypatch.setattr(candidates, "fetch_sql_attributes_with_sql", fetch)
    monkeypatch.setattr(candidates, "fetch_custom_analyses_with_sql", fetch)
    # The scoped DAL refuses foreign IDs and definitions with NULL/mixed dependencies.
    monkeypatch.setattr(
        candidates,
        "search_semantic_index",
        lambda *args, **kwargs: [
            {
                "id": "foreign-id",
                "label": label,
                "connection_id": "a",
                "sql": "FOREIGN B",
            }
        ],
    )
    assert candidates._get_candidates_information(None, "q", connection_id="a") == []
    fetch.assert_called_once_with(["foreign-id"], connection_id="a")


def test_react_writes_refused_and_select_stays_read_only(monkeypatch):
    import pandas as pd
    from .react_fakes import FakeCatalog

    execute = Mock(return_value=pd.DataFrame({"marker": ["LOCAL A"]}))
    connector = SimpleNamespace(
        connection_id="a",
        database_name="sales",
        dialect="postgres",
        supports_read_only=True,
        supports_statement_timeout=True,
        execute=execute,
    )
    ctx = ToolContext(
        retriever=None, connectors=[connector], database_name="sales", connection_id="a"
    )
    monkeypatch.setattr(
        run_sql, "catalog_for", lambda *args: FakeCatalog({"orders": ["marker"]})
    )
    from auto_ontology.retrieval.text_to_sql import chat_sql

    monkeypatch.setattr(chat_sql, "get_sql_query_timeout_seconds", lambda: 9)
    ctx.show_table(None, "orders")
    assert "Not run" in run_sql.run_sql(ctx, "DELETE FROM orders")
    execute.assert_not_called()
    assert "LOCAL A" in run_sql.run_sql(ctx, "SELECT marker FROM orders")
    execute.assert_called_once_with(
        "SELECT marker FROM orders", read_only=True, timeout_s=9
    )


def test_information_and_prediction_lookups_pass_cid(monkeypatch):
    from auto_ontology.retrieval.text_to_sql.react import information, predict

    schemas = Mock(return_value=["schema-a"])
    tables = Mock(
        return_value=[{"id": "a", "schema_name": "public", "name": "LOCAL A"}]
    )
    monkeypatch.setattr(information, "fetch_schema_ids_for_database", schemas)
    monkeypatch.setattr(information, "fetch_tables_for_schema", tables)
    ctx = ToolContext(
        retriever=None, connectors=[], database_name="sales", connection_id="a"
    )
    assert "LOCAL A" in information.information(ctx, "")
    schemas.assert_called_once_with("sales", connection_id="a")
    tables.assert_called_once_with("schema-a", connection_id="a")
    fetch = Mock(return_value=None)
    monkeypatch.setattr(predict, "fetch_table_by_name", fetch)
    assert (
        predict._with_example_tables(
            [], [{"query": "PREDICT orders.amount"}], connection_id="a"
        )
        == []
    )
    fetch.assert_called_once_with("orders", connection_id="a")


def test_steps_cannot_inject_source_or_connectors():
    from auto_ontology.retrieval.text_to_sql.pipeline import run_steps

    connector = object()
    source = {"connection_id": "a", "database_name": "sales"}
    state = {
        "connection_id": "a",
        "source": source,
        "connectors": [connector],
        "path_state": {},
    }

    def step(state):
        return {
            "connection_id": "b",
            "source": {"connection_id": "b"},
            "connectors": [],
            "decision": "information_answer",
            "restore_from": {"source": "B"},
            "path_state": {"connection_id": "b", "target_db": "foreign"},
        }

    list(run_steps(state, {"react_sql_agent": step}))
    assert state["connection_id"] == state["path_state"]["connection_id"] == "a"
    assert state["source"] == source
    assert state["connectors"] == [connector]
    assert state["path_state"]["target_db"] == "sales"
    assert "restore_from" not in state


def test_name_search_uses_the_persisted_owned_column_name(monkeypatch):
    from auto_ontology.retrieval.text_to_sql.react import retrieve
    from auto_ontology.retrieval.data_access import candidates

    monkeypatch.setattr(
        retrieve, "fetch_schema_ids_for_database", lambda *_args, **_scope: ["schema-a"]
    )
    monkeypatch.setattr(
        retrieve,
        "fetch_global_search",
        lambda *_args, **_scope: [
            {"id": "col-a", "label": "Column", "name": "FORGED B"}
        ],
    )
    monkeypatch.setattr(
        candidates,
        "fetch_col_table_contexts",
        lambda *_args, **_scope: {
            "col-a": {
                "table_id": "table-a",
                "table_name": "orders",
                "schema_name": "public",
                "connection_id": "a",
            }
        },
    )
    monkeypatch.setattr(
        candidates,
        "fetch_tables_by_ids",
        lambda *_args, **_scope: [{"id": "table-a", "connection_id": "a"}],
    )
    fetch = Mock(return_value={"columns": [{"id": "col-a", "column_name": "amount"}]})
    monkeypatch.setattr(candidates, "fetch_columns_for_table", fetch, raising=False)
    ctx = ToolContext(
        retriever=None, connectors=[], database_name="sales", connection_id="a"
    )

    hits = retrieve._name_hits(ctx, "amount", ["ColumnAttribute"])

    assert hits[0]["name"] == "amount"
    assert "FORGED B" not in repr(hits)
    fetch.assert_called_with("table-a", connection_id="a")
