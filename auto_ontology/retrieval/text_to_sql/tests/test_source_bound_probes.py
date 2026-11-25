# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The retired preparation/value-probe guards, exercised through ReAct tools."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.text_to_sql.react import catalog as retrieve, run_sql
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from .react_fakes import FakeCatalog


@pytest.mark.parametrize("hit_id", ["foreign-b", "local-a"])
def test_extra_tables_use_only_persisted_source_context(monkeypatch, hit_id):
    persisted = {
        "id": "local-a",
        "connection_id": "a",
        "name": "orders",
        "schema_name": "public",
        "database_name": "sales",
        "description": "LOCAL A",
        "columns": [{"name": "status"}],
    }
    forged = {**persisted, "id": hit_id, "description": "FOREIGN B"}

    def fetch(ids, *, connection_id):
        assert connection_id == "a"
        return [persisted] if "local-a" in ids else []

    monkeypatch.setattr(retrieve, "fetch_tables_by_ids", fetch)
    monkeypatch.setattr(
        retrieve, "_meaning_hits", lambda *args: [{**forged, "label": "Table"}]
    )
    monkeypatch.setattr(retrieve, "_uncovered_physical_hits", lambda *args: [])
    monkeypatch.setattr(retrieve, "_name_hits", lambda *args: [])
    monkeypatch.setattr(retrieve.config, "siblings_enabled", lambda: False)
    ctx = ToolContext(connectors=[], database_name="sales", connection_id="a")
    result = retrieve.retrieve(ctx, "orders")
    assert "FOREIGN B" not in result
    assert ("LOCAL A" in result) == (hit_id == "local-a")


@pytest.mark.parametrize(
    "table",
    [
        "catalog_b.public.orders",
        "catalog_a.other.orders",
        "catalog_a.public.orders",
        "public.orders",
        "orders",
        '"CATALOG_A"."PUBLIC"."ORDERS"',
        "(WITH local_orders AS (SELECT * FROM catalog_a.public.orders) SELECT * FROM local_orders)",
    ],
)
def test_model_probe_cannot_escape_selected_catalog_or_schema(monkeypatch, table):
    allowed = "catalog_b" not in table and ".other." not in table
    connector = SimpleNamespace(
        connection_id="a",
        database_name="catalog_a",
        dialect="trino",
        qualify=lambda s, t: f"catalog_a.{s}.{t}",
    )
    catalog = FakeCatalog({"orders": ["status"]}, {"orders": ["public"]})
    # Unlike the generic fake, this snapshot rejects tables in an unknown schema.
    columns = catalog.columns
    catalog.columns = lambda t, s: (
        columns(t, s) if not s or s.lower() == "public" else None
    )
    monkeypatch.setattr(run_sql, "catalog_for", lambda *args: catalog)
    executor = Mock()
    monkeypatch.setattr(run_sql, "execute_chat_sql", executor)
    ctx = ToolContext(
        connectors=[connector],
        database_name="catalog_a",
        connection_id="a",
    )
    ctx.show_table("public", "orders")
    check = run_sql.check_query(ctx, connector, f"SELECT status FROM {table}")
    assert bool(check.problems) != allowed
    if not allowed:
        result = run_sql.run_sql(ctx, f"SELECT status FROM {table}")
        assert "Not run" in result
        executor.assert_not_called()


def test_ambiguous_probe_is_not_in_view():
    import sqlglot

    ctx = ToolContext(connectors=[], connection_id="a")
    ctx.show_table("public", "orders")
    catalog = FakeCatalog({"orders": ["status"]}, {"orders": ["public", "private"]})
    assert run_sql.tables_not_shown(
        sqlglot.parse_one("SELECT * FROM orders"), ctx, catalog
    )


def test_foreign_legacy_cached_results_are_not_episode_memory(monkeypatch):
    from auto_ontology.retrieval.text_to_sql.react.agent import ReactSQLAgent
    from auto_ontology.retrieval.text_to_sql.react.loop import EpisodeResult
    from auto_ontology.retrieval.text_to_sql.react import agent

    def episode(llm, ctx, **kwargs):
        assert ctx.pages == []
        assert ctx.schemas == {}
        assert ctx.seen_tables == {}
        return EpisodeResult(None, "test", 0)

    monkeypatch.setattr(agent, "run_episode", episode)
    ReactSQLAgent().execute(
        {
            "connection_id": "a",
            "llm": None,
            "initial_question": "q",
            "connectors": [],
            "sql_value_validation_cache": {"FOREIGN B": "rows"},
            "path_state": {"target_db": "sales"},
        }
    )
