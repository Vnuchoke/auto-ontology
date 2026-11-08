# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import (
    candidates_preparation as preparation,
)
from auto_ontology.retrieval.text_to_sql.agents import (
    sql_value_validation as validation,
)


@pytest.mark.parametrize("hit_id", ["foreign-b", "local-a"])
def test_extra_tables_use_only_persisted_source_context(
    monkeypatch: pytest.MonkeyPatch, hit_id: str
) -> None:
    persisted = {
        "id": "local-a",
        "connection_id": "a",
        "name": "orders",
        "database_name": "sales",
        "schema_name": "public",
        "description": "LOCAL A",
        "columns": [
            {
                "name": "status",
                "sample_values": ["LOCAL A"],
                "is_nullable": False,
                "semantic_fk": None,
            }
        ],
    }
    forged = {
        **persisted,
        "id": hit_id,
        "description": "FOREIGN B",
        "columns": [
            {
                "name": "foreign_b",
                "sample_values": ["FOREIGN B"],
                "is_nullable": False,
                "semantic_fk": None,
            }
        ],
    }
    reads: list[list[str]] = []

    def fetch(ids: list[str], *, connection_id: str) -> list[dict]:
        assert connection_id == "a"
        reads.append(ids)
        return [persisted] if "local-a" in ids else []

    def model(
        self: object, state: dict, question: str, tables: list[dict], *args: Any
    ) -> tuple[list[dict], str]:
        assert "FOREIGN B" not in str(tables)
        assert tables == ([persisted] if hit_id == "local-a" else [])
        return tables, ""

    monkeypatch.setattr(
        preparation, "get_relevant_tables", lambda *args, **kwargs: [forged]
    )
    monkeypatch.setattr(preparation, "fetch_tables_by_ids", fetch)
    monkeypatch.setattr(
        preparation, "fetch_custom_analyses_with_sql", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        preparation, "get_relevant_tables_from_candidates", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        preparation, "find_connected_junction_tables", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(
        preparation, "find_kept_table_bridges", lambda *args, **kwargs: ([], [], 0)
    )
    monkeypatch.setattr(
        preparation.CandidatePreparationAgent, "_filter_tables_by_relevance", model
    )
    result = preparation.CandidatePreparationAgent().execute(
        {
            "connection_id": "a",
            "initial_question": "q",
            "data_retriever": object(),
            "path_state": {"connection_id": "a", "target_db": "sales"},
        }
    )
    assert reads == [[hit_id]]
    assert result["path_state"]["relevant_tables"] == (
        [persisted] if hit_id == "local-a" else []
    )


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
def test_model_probe_cannot_escape_selected_catalog_or_schema(
    monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    allowed = "catalog_b" not in table and ".other." not in table
    connector = SimpleNamespace(
        connection_id="a", database_name="catalog_a", dialect="trino"
    )
    executed: list[str] = []
    prompts: list[str] = []
    probe = f"SELECT status FROM {table}"

    class Executor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.budget_left = 10

        def run(self, sql: str, **kwargs: Any) -> dict:
            executed.append(sql)
            return {
                "ok": True,
                "rows": [{"status": "FOREIGN B" if not allowed else "LOCAL A"}],
            }

    def model(
        llm: object, messages: list, schema: object
    ) -> validation.ValueValidationAction:
        prompts.append(str(messages))
        if len(prompts) == 1:
            return validation.ValueValidationAction(
                action="query_database",
                arguments=validation.ValueQueryArguments(
                    sql=probe, purpose="check status"
                ),
            )
        return validation.ValueValidationAction(action="finish")

    monkeypatch.setattr(validation, "ProbeExecutor", Executor)
    monkeypatch.setattr(validation, "invoke_with_structured_output", model)
    result = validation.SQLValueValidationAgent().execute(
        {
            "connection_id": "a",
            "llm": object(),
            "validate_sql_values": True,
            "initial_question": "status",
            "path_state": {
                "sql_code": "SELECT status FROM catalog_a.public.orders WHERE status = 'active'",
                "relevant_tables": [
                    {
                        "id": "local-a",
                        "connection_id": "a",
                        "name": "orders",
                        "database_name": "catalog_a",
                        "schema_name": "public",
                    }
                ],
            },
            "connectors": [connector],
        }
    )
    assert executed == ([probe] if allowed else [])
    assert "FOREIGN B" not in str(prompts)
    assert "FOREIGN B" not in str(result)


def test_probe_identity_resolution_is_unambiguous_and_quote_aware() -> None:
    tables = [
        {"name": "orders", "database_name": "sales", "schema_name": schema}
        for schema in ("public", "private")
    ]
    assert not validation._allowed_probe_tables(
        "SELECT * FROM orders", tables, "postgres"
    )
    assert validation._allowed_probe_tables(
        "SELECT * FROM public.orders", tables, "postgres"
    )
    mixed_case = [{"name": "Orders", "database_name": "sales", "schema_name": "public"}]
    assert validation._allowed_probe_tables(
        'SELECT * FROM public."Orders"', mixed_case, "postgres"
    )
    assert not validation._allowed_probe_tables(
        "SELECT * FROM public.orders", mixed_case, "postgres"
    )
    assert not validation._allowed_probe_tables(
        "WITH orders AS (SELECT 1) SELECT * FROM catalog_b.public.orders",
        tables,
        "trino",
    )


def test_cached_foreign_probe_is_rejected_before_model_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = "SELECT status FROM catalog_b.public.orders"
    executor = SimpleNamespace(budget_left=10, run=Mock())
    prompts: list[str] = []

    def model(
        llm: object, messages: list, schema: object
    ) -> validation.ValueValidationAction:
        prompts.append(str(messages))
        if len(prompts) == 1:
            return validation.ValueValidationAction(
                action="query_database",
                arguments=validation.ValueQueryArguments(sql=probe, purpose="check"),
            )
        return validation.ValueValidationAction(action="finish")

    monkeypatch.setattr(validation, "ProbeExecutor", lambda *args, **kwargs: executor)
    monkeypatch.setattr(validation, "invoke_with_structured_output", model)
    validation.SQLValueValidationAgent().execute(
        {
            "connection_id": "a",
            "initial_question": "status",
            "llm": object(),
            "validate_sql_values": True,
            "connectors": [
                SimpleNamespace(
                    connection_id="a", database_name="catalog_a", dialect="trino"
                )
            ],
            "path_state": {
                "sql_code": "SELECT status FROM orders WHERE status = 'active'",
                "relevant_tables": [
                    {
                        "connection_id": "a",
                        "name": "orders",
                        "database_name": "catalog_a",
                        "schema_name": "public",
                    }
                ],
            },
            "sql_value_validation_cache": {
                "queries": {
                    f"catalog_a:trino::{probe}": {
                        "ok": True,
                        "rows": [{"status": "FOREIGN B"}],
                    }
                }
            },
        }
    )
    executor.run.assert_not_called()
    assert len(prompts) == 2
    assert "FOREIGN B" not in str(prompts)
    assert "outside the relevant schema" in prompts[1]
