# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.text_to_sql.agents import candidates_retrieval as module
from auto_ontology.retrieval.text_to_sql import source_hits


@pytest.fixture(autouse=True)
def owned_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    def rows(ids: list[str], *, connection_id: str) -> list[dict]:
        assert connection_id == "a"
        return [{"id": node_id} for node_id in ids if node_id != "foreign"]

    monkeypatch.setattr(source_hits, "fetch_custom_analyses_with_sql", rows)
    monkeypatch.setattr(source_hits, "fetch_sql_attributes_with_sql", rows)
    monkeypatch.setattr(
        source_hits,
        "fetch_attr_column_contexts",
        lambda ids, *, connection_id: {
            row["id"]: row for row in rows(ids, connection_id=connection_id)
        },
    )
    monkeypatch.setattr(
        source_hits,
        "get_full_term_by_id",
        lambda node_id, *, connection_id: (
            {"id": node_id} if node_id != "foreign" else None
        ),
    )


def test_retrieval_requires_resolved_source_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = Mock()
    monkeypatch.setattr(module, "search_semantic_index", search)
    with pytest.raises(KeyError, match="connection_id"):
        module.CandidateRetrievalAgent().execute({"llm": object(), "path_state": {}})
    search.assert_not_called()


@pytest.mark.parametrize("evidence", ["", "status means active"])
def test_every_search_is_cid_scoped_and_foreign_hits_never_reach_model(
    monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    calls: list[tuple[str, dict]] = []

    def search(retriever: object, query: str, **kwargs: object) -> list[dict]:
        calls.append((query, kwargs))
        label = kwargs["label_filter"][0]
        return [
            {"id": label, "connection_id": "a", "database_name": "sales", "score": 0.2},
            {
                "id": "foreign",
                "connection_id": "a",
                "database_name": "sales",
                "text": "FOREIGN B SQL",
                "score": 0.01,
            },
        ]

    def filter_hits(
        llm: object, question: str, custom: list[dict], attrs: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        assert all(hit["connection_id"] == "a" for hit in custom + attrs)
        assert all(hit["id"] != "foreign" for hit in custom + attrs)
        return custom, attrs

    monkeypatch.setattr(module, "search_semantic_index", search)
    monkeypatch.setattr(
        module, "custom_analysis_exists", lambda *, connection_id: connection_id == "a"
    )
    monkeypatch.setattr(module, "_llm_filter_both", filter_hits)
    result = module.CandidateRetrievalAgent().execute(
        {
            "connection_id": "a",
            "llm": object(),
            "initial_question": "revenue",
            "evidence": evidence,
            "semantic_retriever": object(),
            "path_state": {
                "connection_id": "a",
                "target_db": "sales",
                "entities": ["revenue"],
                "subject": "orders",
            },
        }
    )
    assert calls
    assert all(kwargs["connection_id"] == "a" for _, kwargs in calls)
    assert (evidence in [query for query, _ in calls]) == bool(evidence)
    assert result["path_state"]["target_db"] == "sales"
    assert result["path_state"]["retrieved_subject_term"]["connection_id"] == "a"
    for key in (
        "retrieved_column_attributes",
        "retrieved_custom_analyses",
        "retrieved_sql_attributes",
    ):
        assert all(hit["connection_id"] == "a" for hit in result["path_state"][key])
        assert all(hit["id"] != "foreign" for hit in result["path_state"][key])


def test_bridge_attributes_skip_combined_model_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.semantic.constants import SQL_ATTR_SOURCE_BRIDGE

    monkeypatch.setattr(module, "custom_analysis_exists", lambda **kwargs: False)
    monkeypatch.setattr(
        module,
        "search_semantic_index",
        lambda *args, **kwargs: [
            {"id": "bridge", "connection_id": "a", "source": SQL_ATTR_SOURCE_BRIDGE}
        ],
    )
    combined = Mock(side_effect=AssertionError("bridge should not invoke model"))
    monkeypatch.setattr(module, "_llm_filter_both", combined)
    module.CandidateRetrievalAgent().execute(
        {
            "connection_id": "a",
            "llm": object(),
            "initial_question": "q",
            "semantic_retriever": object(),
            "path_state": {"target_db": "sales"},
        }
    )
    combined.assert_not_called()
