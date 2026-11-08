# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock, patch

import pytest

from auto_ontology.connectors.sources import SourceDescriptor, SourceSelectionError
from auto_ontology.retrieval.interactive import clarify, coordinator, entity_resolution
from auto_ontology.retrieval.interactive.state import InteractiveSessionState

A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[SourceDescriptor]:
    active = [
        SourceDescriptor(cid, cid, "Sales", "postgres", "managed") for cid in (A, B)
    ]
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources", lambda: active
    )
    monkeypatch.setattr(
        "auto_ontology.retrieval.source_context.semantic_layer_calculated",
        lambda **_: True,
    )
    monkeypatch.setattr(
        "auto_ontology.retrieval.source_context.require_source_vectors_ready",
        lambda _: None,
    )
    return active


def create(**kwargs: Any) -> InteractiveSessionState:
    db_name = kwargs.pop("db_name", None)
    return coordinator.create_session(
        session_id="s",
        task_id="t",
        db_name=db_name,
        db_schema="",
        external_kb="",
        question="sales?",
        data_retriever=None,
        semantic_retriever=object(),
        connectors=[
            SimpleNamespace(connection_id=B, database_name="Sales"),
            SimpleNamespace(connection_id=A, database_name="Sales"),
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "selector,code",
    [
        ({}, "source_required"),
        ({"connection_id": B, "connection_label": "B"}, "invalid_source_selector"),
    ],
)
def test_selection_fails_before_models_or_search(
    sources: list[SourceDescriptor], selector: dict[str, str], code: str
) -> None:
    with (
        patch.object(coordinator, "_get_fast_llm") as model,
        patch.object(entity_resolution, "search_semantic_index") as search,
    ):
        with pytest.raises(SourceSelectionError, match=code):
            create(**selector)
    model.assert_not_called()
    search.assert_not_called()


def test_selected_identity_is_separate_from_physical_name(
    sources: list[SourceDescriptor],
) -> None:
    session = create(connection_id=A)
    assert session.connection_id == A
    assert session.db_name == "Sales"
    assert session.path_state == {"connection_id": A, "task_id": "t"}
    assert [c.connection_id for c in session.connectors] == [A]
    with pytest.raises((AttributeError, ValueError)):
        session.connection_id = B


def test_direct_init_requires_readiness(
    sources: list[SourceDescriptor], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "auto_ontology.retrieval.source_context.semantic_layer_calculated",
        lambda **_: False,
    )
    with pytest.raises(SourceSelectionError, match="source_not_ready"):
        create(connection_id=A)


def test_no_sources_is_unavailable(sources: list[SourceDescriptor]) -> None:
    sources.clear()
    with pytest.raises(SourceSelectionError, match="source_unavailable") as error:
        create()
    assert error.value.status_code == 503


def test_one_source_is_automatic(sources: list[SourceDescriptor]) -> None:
    sources.pop()
    assert create().connection_id == A
    assert create(db_name="Sales").connection_id == A


def test_legacy_physical_name_is_ambiguous(sources: list[SourceDescriptor]) -> None:
    with pytest.raises(SourceSelectionError, match="ambiguous_source") as error:
        create(db_name="Sales")
    assert error.value.status_code == 409


def test_canonical_and_legacy_selectors_are_exclusive(
    sources: list[SourceDescriptor],
) -> None:
    with pytest.raises(SourceSelectionError, match="invalid_source_selector"):
        create(connection_id=A, db_name="Sales")


def test_connector_routing_ignores_foreign_resumed_tables(
    sources: list[SourceDescriptor],
) -> None:
    session = create(connection_id=A)
    session.path_state["relevant_tables"] = [
        {"connection_id": B, "database_name": "Sales"}
    ]
    assert coordinator._session_connector(session).connection_id == A


def test_ambiguity_requeries_use_canonical_identity() -> None:
    with patch.object(
        entity_resolution, "search_semantic_index", return_value=[]
    ) as search:
        entity_resolution._collect_ambiguous_hits(
            object(), "Sales", {"sales", "revenue"}, connection_id=A
        )
    assert search.call_count == 2
    assert all(call.kwargs["connection_id"] == A for call in search.call_args_list)


@pytest.mark.parametrize("kb_formula", [False, True])
def test_collision_requeries_use_canonical_identity(kb_formula: bool) -> None:
    shared: dict[str, Any] = {
        "id": "Sales.public.orders.amount",
        "score": 0.1,
        "connection_id": B,
    }
    hits = {"sales": dict(shared), "revenue": dict(shared)}
    kwargs: dict[str, Any] = (
        {
            "kb_covered_norms": {"sales"},
            "kb_text_by_norm": {"sales": ("Sales", "SUM(amount)")},
        }
        if kb_formula
        else {}
    )
    with (
        patch.object(
            entity_resolution, "search_semantic_index", return_value=[]
        ) as search,
        patch.object(
            entity_resolution, "_llm_disambiguate_collision", return_value=None
        ),
    ):
        entity_resolution._resolve_collisions(
            "sales?", "", hits, object(), "Sales", connection_id=A, **kwargs
        )
    assert search.call_count == 2
    assert all(call.kwargs["connection_id"] == A for call in search.call_args_list)


def test_coverage_graph_receives_canonical_identity() -> None:
    with (
        patch.object(entity_resolution, "get_llm_client", return_value=object()),
        patch.object(
            entity_resolution._ec_app, "invoke", return_value={"path_state": {}}
        ) as invoke,
    ):
        entity_resolution._run_entity_coverage_pipeline(
            "sales", object(), "Sales", connection_id=A
        )
    state = invoke.call_args.args[0]
    assert state["connection_id"] == state["path_state"]["connection_id"] == A
    assert "target_db" not in state["path_state"]


def test_missing_identity_rejected_before_search() -> None:
    with patch.object(entity_resolution, "search_semantic_index") as search:
        with pytest.raises(SourceSelectionError, match="source_required"):
            entity_resolution._entity_ranked_hits("sales", object(), "Sales")
    search.assert_not_called()


def test_missing_identity_rejected_before_coverage_model() -> None:
    with patch.object(entity_resolution, "get_llm_client") as model:
        with pytest.raises(SourceSelectionError, match="source_required"):
            entity_resolution._run_entity_coverage_pipeline("sales", object(), "Sales")
    model.assert_not_called()


def test_sql_payload_and_resume_cannot_replace_identity(
    sources: list[SourceDescriptor],
) -> None:
    import sys
    import types

    session = create(connection_id=A)
    session.path_state.update(
        connection_id=B, target_db="Sales", _resume_from="reconstruct_sql"
    )
    session._grounded_knowledge_for = session.working_question
    agent = Mock(
        return_value={
            "sql_code": "SELECT 1",
            "path_state": {"connection_id": B, "target_db": "Sales"},
        }
    )
    main = types.ModuleType("auto_ontology.retrieval.text_to_sql.main")
    main.get_agent_response_with_state = agent
    state = types.ModuleType("auto_ontology.retrieval.text_to_sql.state")
    state.TextToSQLPayload = dict
    with (
        patch.dict(sys.modules, {main.__name__: main, state.__name__: state}),
        patch.object(coordinator, "generate_evidence", return_value=""),
    ):
        assert coordinator._run_sql_generation(session) == "SELECT 1"
    payload = agent.call_args.args[0]
    assert payload["connection_id"] == payload["path_state"]["connection_id"] == A
    assert "target_db" not in payload and "target_db" not in payload["path_state"]
    assert session.path_state["connection_id"] == A
    assert "target_db" not in session.path_state
    coordinator._apply_follow_up_seed(session, follow_up_question="and revenue?")
    assert session.path_state["connection_id"] == A


@pytest.mark.parametrize("entry", ["should_clarify", "refresh_grounded_knowledge"])
def test_clarification_forwards_session_identity_and_preserves_kb_cache(
    sources: list[SourceDescriptor],
    entry: Literal["should_clarify", "refresh_grounded_knowledge"],
) -> None:
    session = create(connection_id=A)
    session.cumulative_grounded_knowledge = "prior KB"
    grounded_knowledge = "Revenue = SUM(amount)"
    coverage_result = ([], [], grounded_knowledge, set(), {}, set(), [])
    with (
        patch.object(
            clarify, "_find_unresolvable_entities", return_value=coverage_result
        ) as coverage,
        patch.object(clarify, "safe_invoke_text", return_value="PROCEED"),
        patch(
            "auto_ontology.retrieval.interactive.completeness.detect_incomplete_formulas",
            return_value=[],
        ),
    ):
        for _ in range(2):
            if entry == "should_clarify":
                assert clarify.should_clarify(session, Mock()) == (False, None)
            else:
                clarify.refresh_grounded_knowledge(session)
    coverage.assert_called_once()
    assert coverage.call_args.kwargs.get("connection_id") == A
    assert coverage.call_args.args[2] == "Sales"
    assert session._grounded_knowledge == grounded_knowledge
    assert session._grounded_knowledge_for == session.working_question
    assert session._cached_unresolvable_for == session.working_question
    assert "prior KB" in session.cumulative_grounded_knowledge
