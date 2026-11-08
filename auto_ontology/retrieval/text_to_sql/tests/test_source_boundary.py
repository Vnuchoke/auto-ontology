# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.connectors.sources import SourceDescriptor, SourceSelectionError
from auto_ontology.retrieval.text_to_sql import main


@pytest.fixture
def source_gate(monkeypatch: pytest.MonkeyPatch) -> Mock:
    import sys

    gate = Mock(return_value=SourceDescriptor("a", "Local", "sales", "postgres", "env"))
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=gate),
    )
    return gate


def test_boundary_selects_only_exact_connection_and_overwrites_resume(
    source_gate: Mock,
) -> None:
    first = SimpleNamespace(
        connection_id="a", database_name="sales", dialect="postgres"
    )
    other = SimpleNamespace(
        connection_id="b", database_name="sales", dialect="postgres"
    )
    state = main._build_state(
        {
            "question": "q",
            "connection_id": "a",
            "connectors": [other, first],
            "data_retriever": object(),
            "path_state": {"connection_id": "b", "target_db": "foreign"},
        }
    )
    assert state["connectors"] == [first]
    assert state["connection_id"] == state["path_state"]["connection_id"] == "a"
    assert state["path_state"]["target_db"] == "sales"
    assert state["source"] == {
        "connection_id": "a",
        "display_label": "Local",
        "database_name": "sales",
    }


def test_foreign_resume_cannot_reuse_compiled_sql_context(source_gate: Mock) -> None:
    state = main._build_state(
        {
            "question": "q",
            "connection_id": "a",
            "connectors": [SimpleNamespace(connection_id="a", database_name="sales")],
            "data_retriever": object(),
            "path_state": {
                "connection_id": "b",
                "_resume_from": "construct_sql_from_candidates",
                "sql_code": "SELECT foreign_b",
                "custom_analyses_str": "FOREIGN B",
                "term_synonyms": {"foreign": ["B"]},
            },
        }
    )
    for key in ("_resume_from", "sql_code", "custom_analyses_str", "term_synonyms"):
        assert key not in state["path_state"]


def test_unknown_source_fails_before_graph_or_payload_checks(
    source_gate: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_gate.side_effect = SourceSelectionError(422, "source_required")
    graph = Mock()
    monkeypatch.setattr(main, "app", graph)
    with pytest.raises(SourceSelectionError, match="source_required"):
        main.get_agent_response({"question": "q"})
    graph.stream.assert_not_called()


def test_merge_and_answer_cannot_override_source() -> None:
    source = {"connection_id": "a", "display_label": "Local", "database_name": "sales"}
    state = {
        "connection_id": "a",
        "source": source,
        "path_state": {"connection_id": "a", "target_db": "sales"},
    }
    main._merge_node_output(
        state,
        {
            "connection_id": "b",
            "source": {"connection_id": "b"},
            "path_state": {
                "connection_id": "b",
                "target_db": "foreign",
                "final_response": {"source": {"connection_id": "b"}},
            },
        },
    )
    assert state["connection_id"] == state["path_state"]["connection_id"] == "a"
    assert state["path_state"]["target_db"] == "sales"
    assert main._extract_answer(state)["source"] == source


@pytest.mark.parametrize("connection_id", ["a", "b"])
def test_same_sales_connections_execute_distinct_results(
    source_gate: Mock, monkeypatch: pytest.MonkeyPatch, connection_id: str
) -> None:
    import json
    import pandas as pd
    from auto_ontology.retrieval.text_to_sql.agents import sql_execution

    source_gate.return_value = SourceDescriptor(
        connection_id, connection_id.upper(), "sales", "postgres", "env"
    )
    connectors = [
        SimpleNamespace(connection_id=cid, database_name="sales", dialect="postgres")
        for cid in ("a", "b")
    ]
    executed: list[str] = []

    def execute(connector: object, sql: str) -> pd.DataFrame:
        executed.append(connector.connection_id)
        return pd.DataFrame({"marker": [connector.connection_id]})

    monkeypatch.setattr(sql_execution, "execute_chat_sql", execute)
    state = main._build_state(
        {
            "question": "q",
            "connection_id": connection_id,
            "connectors": connectors,
            "data_retriever": object(),
            "path_state": {
                "connection_id": connection_id,
                "sql_code": "SELECT marker FROM public.orders",
                "relevant_tables": [
                    {
                        "id": f"orders-{connection_id}",
                        "connection_id": connection_id,
                        "database_name": "sales",
                        "name": "orders",
                    }
                ],
            },
        }
    )
    result = sql_execution.SQLExecutionAgent().execute(state)
    assert executed == [connection_id]
    assert json.loads(result["path_state"]["sql_response_from_db"][0]) == [
        {"marker": connection_id}
    ]


def test_foreign_resume_dependencies_rejected_before_graph(
    source_gate: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = Mock()
    monkeypatch.setattr(main, "app", graph)
    with pytest.raises(SourceSelectionError, match="source_conflict"):
        main.get_agent_response(
            {
                "question": "q",
                "connection_id": "a",
                "connectors": [
                    SimpleNamespace(connection_id="a", database_name="sales")
                ],
                "data_retriever": object(),
                "path_state": {
                    "relevant_tables": [
                        {"connection_id": "b", "database_name": "sales"}
                    ]
                },
            }
        )
    graph.stream.assert_not_called()
