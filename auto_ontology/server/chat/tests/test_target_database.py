# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from auto_ontology.connectors import sources
from auto_ontology.connectors.databricks_oauth import DatabricksOAuthError
from auto_ontology.server.chat import router
from auto_ontology.server.chat.helpers import ChatRequest


SOURCE = sources.SourceDescriptor("opaque-A", "East", "sales", "postgres", "managed")
SNAPSHOT = {
    "connection_id": "opaque-A",
    "display_label": "East",
    "database_name": "sales",
}


def http_request(headers: Sequence[tuple[bytes, bytes]] = ()) -> Request:
    return Request({"type": "http", "headers": list(headers)})


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # Readiness belongs to the shared retrieval module, not the HTTP boundary.
    resolve = Mock(return_value=SOURCE)
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=resolve),
    )
    return resolve


@pytest.mark.parametrize(
    "status,code",
    [
        (422, "invalid_source_selector"),
        (422, "source_required"),
        (404, "source_unavailable"),
        (409, "ambiguous_source"),
        (503, "source_not_ready"),
    ],
)
def test_selection_precedes_every_side_effect(
    boundary: Mock,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    code: str,
) -> None:
    from auto_ontology.connectors import registry

    boundary.side_effect = sources.SourceSelectionError(status, code)
    spies = []
    for name in (
        "prepare_conversation",
        "create_stateless_analytics",
        "get_pool",
        "_subject_token",
        "resolve_internal_user",
        "get_llm_client",
        "get_non_reasoning_llm_client",
    ):
        spy = Mock(side_effect=AssertionError(name))
        monkeypatch.setattr(router, name, spy)
        spies.append(spy)
    for name in ("get_connectors", "get_connectors_for_subject_token"):
        spy = Mock(side_effect=AssertionError(name))
        monkeypatch.setattr(registry, name, spy)
        spies.append(spy)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(
                    question="orders",
                    connection_id="bad",
                ),
                http_request(),
            )
        )
    assert (exc.value.status_code, exc.value.detail) == (status, code)
    for spy in spies:
        spy.assert_not_called()


def test_stateless_first_source_and_authoritative_result_replay(
    boundary: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = Mock()
    worker.events.return_value = iter(
        [
            {
                "type": "result",
                "answer": {"response": "1", "source": {"connection_id": "forged"}},
            }
        ]
    )
    pool = Mock()
    pool.acquire.return_value = worker
    monkeypatch.setattr(router, "get_pool", lambda: pool)
    monkeypatch.setattr(router, "resolve_internal_user", lambda *args, **kwargs: None)
    auth = Mock(return_value="bearer")
    monkeypatch.setattr(router, "_subject_token", auth)

    async def run() -> str:
        response = await router.chat_completions(
            ChatRequest(question="orders", connection_label="East"), http_request()
        )
        return "".join([chunk async for chunk in response.body_iterator])

    stream = asyncio.run(run())
    frames = [
        json.loads(line[6:])
        for line in stream.splitlines()
        if line.startswith("data: {")
    ]
    assert frames[0] == {"type": "source", "source": SNAPSHOT}
    assert frames[1]["answer"]["source"] == SNAPSHOT
    boundary.assert_called_once_with(
        connection_id=None, connection_label="East", target_db=None
    )
    auth.assert_called_once()
    assert auth.call_args.args[1] == "opaque-A"
    assert worker.submit.call_args.kwargs["connection_id"] == "opaque-A"
    assert not worker.submit.call_args.kwargs.get("target_db")


@pytest.mark.parametrize("field", ["connection_id", "connection_label", "target_db"])
def test_selector_fields_are_strict(field: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="orders", **{field: 12})


def test_sso_gate_is_selected_only_and_preserves_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.connectors import registry

    selected = Mock(side_effect=DatabricksOAuthError("no token"))
    monkeypatch.setattr(registry, "get_connectors_for_subject_token", selected)
    monkeypatch.setattr(
        registry, "get_connectors", Mock(side_effect=AssertionError("scan"))
    )
    with pytest.raises(HTTPException) as exc:
        router._subject_token(http_request(), "opaque-A")
    assert exc.value.status_code == 401
    selected.assert_called_once_with(None, connection_id="opaque-A")
    selected.reset_mock()
    assert (
        router._subject_token(
            http_request([(b"authorization", b"Bearer caller")]), "opaque-A"
        )
        == "caller"
    )
    selected.assert_not_called()


def test_non_sso_selected_source_allows_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.connectors import registry

    selected = Mock(return_value=[Mock()])
    monkeypatch.setattr(registry, "get_connectors_for_subject_token", selected)
    assert router._subject_token(http_request(), "opaque-A") is None
    selected.assert_called_once_with(None, connection_id="opaque-A")


def test_watch_replays_source_snapshot_and_authoritative_answer(
    boundary: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = uuid.uuid4()
    worker = Mock()
    worker.events.return_value = iter(
        [{"type": "result", "answer": {"response": "1", "source": "forged"}}]
    )
    pool = Mock()
    pool.acquire.return_value = worker
    monkeypatch.setattr(router, "get_pool", lambda: pool)
    monkeypatch.setattr(router, "resolve_internal_user", lambda *args, **kwargs: "user")
    monkeypatch.setattr(router, "_subject_token", lambda *args: None)
    monkeypatch.setattr(
        router,
        "prepare_conversation",
        lambda **kwargs: SimpleNamespace(
            analytics_id="analytics", history=[], source=SOURCE
        ),
    )
    monkeypatch.setattr(router, "persist_assistant_result", Mock())
    monkeypatch.setattr(router, "_build_charts_event", lambda *args: None)
    # Keep the completed slot inspectable, as a watch attached during the run is.
    monkeypatch.setattr(router, "_release", lambda slot: None)
    monkeypatch.setattr(router, "_active_slots", {})

    async def run() -> tuple[str, str]:
        response = await router.chat_completions(
            ChatRequest(
                question="orders",
                conversation_id=conversation_id,
                connection_id="opaque-A",
            ),
            http_request(),
        )
        original = "".join([chunk async for chunk in response.body_iterator])
        response = await router.chat_watch(conversation_id, http_request())
        replay = "".join([chunk async for chunk in response.body_iterator])
        return original, replay

    original, replay = asyncio.run(run())
    original_frames = [
        json.loads(line[6:])
        for line in original.splitlines()
        if line.startswith("data: {")
    ]
    replay_frames = [
        json.loads(line[6:])
        for line in replay.splitlines()
        if line.startswith("data: {")
    ]
    assert original_frames == replay_frames
    assert replay_frames[0] == {"type": "source", "source": SNAPSHOT}
    assert replay_frames[1]["answer"]["source"] == SNAPSHOT
