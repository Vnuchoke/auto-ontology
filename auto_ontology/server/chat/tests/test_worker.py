# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the prewarmed chat worker's request loop."""

from typing import Any
from types import SimpleNamespace
from unittest.mock import Mock

from pytest import MonkeyPatch

from auto_ontology.server.chat import worker as worker_module


class _FakeQueue:
    """Queue stub: yields queued messages, then ends the loop like a closed pipe."""

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items = list(items or [])
        self.puts: list[Any] = []

    def get(self) -> Any:
        if self._items:
            return self._items.pop(0)
        # The real loop returns when the pipe closes; this ends the test cleanly.
        raise EOFError

    def put(self, item: Any) -> None:
        self.puts.append(item)


def _run_loop(
    monkeypatch: MonkeyPatch,
    selected_connectors: list[str],
    *,
    evidence: str | None = None,
) -> dict:
    """Drive one ASK; opening the unselected/unavailable neighbor is forbidden."""
    # Importing the agent module builds LLM clients at import time and lets an
    # EnvironmentError escape when no key is configured. Stub the factories before that
    # import is triggered below; the clients are never used, since stream_agent_response
    # is replaced too. (Patching the env instead is not enough — model config resolution
    # is cached, so an earlier import in the same session wins.)
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_llm_client", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "auto_ontology.utils.llm_invoke.get_non_reasoning_llm_client",
        lambda *a, **k: None,
    )

    import sys

    gate = Mock(return_value=SimpleNamespace(connection_id="a"))
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=gate),
    )

    def fake_get_connectors(token: str | None, *, connection_id: str) -> list[str]:
        assert gate.called
        assert connection_id == "a"
        return selected_connectors

    captured: dict = {}

    def fake_stream(payload: dict) -> Any:
        captured.update(payload)
        return iter(())

    monkeypatch.setattr(
        "auto_ontology.connectors.get_connectors",
        Mock(side_effect=AssertionError("must not open all sources")),
    )
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.get_connectors_for_subject_token",
        fake_get_connectors,
    )
    monkeypatch.setattr(
        "auto_ontology.utils.get_data_objects_retriever", lambda: "data"
    )
    monkeypatch.setattr(
        "auto_ontology.utils.get_semantic_objects_retriever", lambda: "semantic"
    )
    monkeypatch.setattr(
        "auto_ontology.server.chat.settings_dal.fetch_acronyms", lambda: []
    )
    monkeypatch.setattr(
        "auto_ontology.server.chat.settings_dal.fetch_custom_prompts", lambda: ""
    )
    monkeypatch.setattr(
        "auto_ontology.retrieval.text_to_sql.main.stream_agent_response", fake_stream
    )

    in_q = _FakeQueue(
        [(worker_module._MSG_ASK, ("q", None, None, None, [], evidence, "a"))]
    )
    worker_module._worker_loop(in_q, _FakeQueue())
    return captured


def test_worker_resolves_selected_connector_before_building_the_payload(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = _run_loop(monkeypatch, selected_connectors=["selected-connector"])

    assert payload["connectors"] == ["selected-connector"]
    assert payload["connection_id"] == "a"


def test_worker_builds_selected_source_payload(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = _run_loop(monkeypatch, selected_connectors=["selected-connector"])

    assert payload["connectors"] == ["selected-connector"]
    assert payload["question"] == "q"
    assert payload["evidence"] == ""


def test_worker_forwards_evidence_without_changing_question(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = _run_loop(
        monkeypatch,
        selected_connectors=["selected-connector"],
        evidence="status means accounts.status",
    )

    assert payload["question"] == "q"
    assert payload["processing_question"] == "q"
    assert payload["evidence"] == "status means accounts.status"


def test_worker_startup_does_not_open_sources_or_vectors(
    monkeypatch: MonkeyPatch,
) -> None:
    import sys

    gate = Mock()
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=gate),
    )
    connectors = Mock(side_effect=AssertionError("unexpected connector"))
    vectors = Mock(side_effect=AssertionError("unexpected VDB"))
    monkeypatch.setattr("auto_ontology.connectors.get_connectors", connectors)
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.get_connectors_for_subject_token", connectors
    )
    monkeypatch.setattr("auto_ontology.utils.get_data_objects_retriever", vectors)
    monkeypatch.setattr("auto_ontology.utils.get_semantic_objects_retriever", vectors)
    output = _FakeQueue()
    worker_module._worker_loop(_FakeQueue(), output)
    assert output.puts == [(worker_module._TAG_READY, None)]
    connectors.assert_not_called()
    vectors.assert_not_called()
    gate.assert_not_called()


def test_source_rejection_precedes_followup_connectors_and_vectors(
    monkeypatch: MonkeyPatch,
) -> None:
    import sys
    from auto_ontology.connectors.sources import SourceSelectionError

    gate = Mock(side_effect=SourceSelectionError(422, "source_required"))
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=gate),
    )
    effects = Mock(side_effect=AssertionError("must reject before model/driver/VDB"))
    monkeypatch.setattr(
        "auto_ontology.retrieval.text_to_sql.follow_up.resolve_follow_up", effects
    )
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.get_connectors_for_subject_token", effects
    )
    monkeypatch.setattr("auto_ontology.utils.get_data_objects_retriever", effects)
    monkeypatch.setattr("auto_ontology.utils.get_semantic_objects_retriever", effects)
    output = _FakeQueue()
    worker_module._worker_loop(
        _FakeQueue(
            [(worker_module._MSG_ASK, ("q", None, None, "token", [], None, None))]
        ),
        output,
    )
    assert (worker_module._TAG_ERROR, "source_required") in output.puts
    effects.assert_not_called()
