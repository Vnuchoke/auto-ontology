# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP orchestration accepts a bound turn before using its source."""

import asyncio
import json
import sys
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from auto_ontology.connectors.sources import SourceSelectionError
from auto_ontology.server.chat import router
from auto_ontology.server.chat.helpers import ChatRequest
from auto_ontology.server.chat.tests.test_target_database import (
    SOURCE,
    SNAPSHOT,
    http_request,
)


@pytest.fixture(name="boundary")
def source_boundary(monkeypatch: pytest.MonkeyPatch) -> Mock:
    resolve = Mock(return_value=SOURCE)
    monkeypatch.setitem(
        sys.modules,
        "auto_ontology.retrieval.source_context",
        SimpleNamespace(resolve_query_source=resolve),
    )
    return resolve


@pytest.fixture
def run_boundary(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock, Mock]:
    worker = Mock()
    worker.events.return_value = iter([])
    pool = Mock()
    pool.acquire.return_value = worker
    monkeypatch.setattr(router, "get_pool", lambda: pool)
    monkeypatch.setattr(router, "_active_slots", {})
    monkeypatch.setattr(router, "resolve_internal_user", lambda *a, **kw: "owner")
    prepare = Mock(
        return_value=SimpleNamespace(source=SOURCE, analytics_id="a", history=[])
    )
    monkeypatch.setattr(router, "prepare_conversation", prepare)
    auth = Mock(return_value=None)
    monkeypatch.setattr(router, "_subject_token", auth)
    return worker, pool, prepare, auth


@pytest.mark.parametrize("label", [None, "request-label-not-authoritative"])
def test_inherited_source_is_authoritative(
    boundary: Mock, run_boundary: tuple[Mock, Mock, Mock, Mock], label: str | None
) -> None:
    worker, pool, prepare, auth = run_boundary
    boundary.side_effect = AssertionError(
        "global selection must not precede inheritance"
    )
    cid = uuid.uuid4()

    def accept(**kwargs: object) -> SimpleNamespace:
        slot = router._active_slots[f"owner:{cid}"]
        assert slot.source == {}
        assert slot.buffer == []
        auth.assert_not_called()
        return SimpleNamespace(source=SOURCE, analytics_id="a", history=[])

    prepare.side_effect = accept

    async def run() -> str:
        response = await router.chat_completions(
            ChatRequest(question="q", conversation_id=cid, connection_label=label),
            http_request(),
        )
        return "".join([chunk async for chunk in response.body_iterator])

    stream = asyncio.run(run())
    assert json.loads(stream.splitlines()[0][6:]) == {
        "type": "source",
        "source": SNAPSHOT,
    }
    assert prepare.call_args.kwargs["connection_id"] is None
    assert prepare.call_args.kwargs["connection_label"] == label
    assert worker.submit.call_args.kwargs["connection_id"] == SOURCE.connection_id
    auth.assert_called_once()
    pool.return_alive.assert_called_once_with(worker)


@pytest.mark.parametrize(
    "status,code",
    [
        (422, "invalid_source_selector"),
        (409, "conversation_source_conflict"),
        (409, "legacy_conversation_read_only"),
        (404, "source_unavailable"),
    ],
)
def test_refused_turn_does_not_authenticate_or_submit(
    boundary: Mock, run_boundary: tuple[Mock, Mock, Mock, Mock], status: int, code: str
) -> None:
    worker, pool, prepare, auth = run_boundary
    boundary.side_effect = AssertionError("global resolution")
    prepare.side_effect = SourceSelectionError(status, code)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(
                    question="q",
                    conversation_id=uuid.uuid4(),
                    connection_label="forged",
                ),
                http_request(),
            )
        )
    assert (exc.value.status_code, exc.value.detail) == (status, code)
    auth.assert_not_called()
    worker.submit.assert_not_called()
    pool.return_alive.assert_called_once_with(worker)


def test_missing_sso_after_acceptance_persists_guarded_error(
    boundary: Mock,
    run_boundary: tuple[Mock, Mock, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, pool, prepare, auth = run_boundary
    boundary.side_effect = AssertionError("global resolution")
    auth.side_effect = HTTPException(401, "SSO required")
    persist = Mock()
    monkeypatch.setattr(router, "persist_assistant_result", persist)
    cid = uuid.uuid4()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(question="q", conversation_id=cid), http_request()
            )
        )
    assert exc.value.status_code == 401
    assert persist.call_args.kwargs["connection_id"] == SOURCE.connection_id
    assert persist.call_args.kwargs["analytics_id"] == "a"
    worker.submit.assert_not_called()
    pool.return_alive.assert_called_once_with(worker)


def test_watch_attached_during_accepted_source_failure_finishes(
    run_boundary: tuple[Mock, Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, pool, prepare, auth = run_boundary
    captured: list[router._Slot] = []
    cid = uuid.uuid4()

    def fail_sso(*args: object) -> None:
        captured.append(router._active_slots[f"owner:{cid}"])
        raise HTTPException(401, "SSO required")

    auth.side_effect = fail_sso
    monkeypatch.setattr(router, "persist_assistant_result", Mock())
    with pytest.raises(HTTPException):
        asyncio.run(
            router.chat_completions(
                ChatRequest(question="q", conversation_id=cid), http_request()
            )
        )
    slot = captured[0]
    assert slot.finished.is_set()
    assert slot.buffer[0] == {"type": "source", "source": SNAPSHOT}
    assert slot.buffer[-1]["type"] == "error"
    assert list(router._stream_slot(slot))[-1] == "data: [DONE]\n\n"


def test_stateless_analytics_does_not_require_conversation_binding(
    run_boundary: tuple[Mock, Mock, Mock, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, pool, prepare, auth = run_boundary
    worker.events.return_value = iter(
        [{"type": "error", "message": "chosen source failed"}]
    )
    analytics = Mock()
    monkeypatch.setattr(router, "persist_analytics_result", analytics)
    slot = router._Slot(
        key="stateless",
        worker=worker,
        user_id="owner",
        conversation_id=None,
        analytics_id="a",
        question="q",
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
    )
    router._pump(slot)
    analytics.assert_called_once_with(
        analytics_id="a",
        user_id="owner",
        response="chosen source failed",
        sql_code=None,
    )
    pool.return_alive.assert_called_once_with(worker)


def test_live_slot_conflict_does_not_accept_another_turn(
    boundary: Mock, run_boundary: tuple[Mock, Mock, Mock, Mock]
) -> None:
    worker, pool, prepare, auth = run_boundary
    cid = uuid.uuid4()
    previous = router._Slot(
        key=f"owner:{cid}",
        worker=Mock(),
        user_id="owner",
        conversation_id=cid,
        analytics_id=None,
        question="previous",
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
    )
    previous.client_alive.set()
    router._active_slots[previous.key] = previous
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(question="q", conversation_id=cid), http_request()
            )
        )
    assert exc.value.status_code == 409
    prepare.assert_not_called()
    auth.assert_not_called()
    worker.submit.assert_not_called()
    pool.return_alive.assert_called_once_with(worker)
    assert router._active_slots[previous.key] is previous


def test_foreign_owner_denial_is_404_without_source_lookup(
    boundary: Mock, run_boundary: tuple[Mock, Mock, Mock, Mock]
) -> None:
    worker, pool, prepare, auth = run_boundary
    prepare.side_effect = router.ConversationAccessError("private")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(question="q", conversation_id=uuid.uuid4()), http_request()
            )
        )
    assert (exc.value.status_code, exc.value.detail) == (404, "Conversation not found")
    boundary.assert_not_called()
    auth.assert_not_called()
    worker.submit.assert_not_called()
    pool.return_alive.assert_called_once_with(worker)


@pytest.mark.parametrize(
    "denial",
    [
        SourceSelectionError(404, "source_unavailable"),
        router.ConversationAccessError("secret://url"),
    ],
)
def test_accepted_sso_error_does_not_retry_denied_write(
    boundary: Mock,
    run_boundary: tuple[Mock, Mock, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    denial: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker, pool, prepare, auth = run_boundary
    auth.side_effect = HTTPException(401, "SSO required")
    persist = Mock(side_effect=denial)
    monkeypatch.setattr(router, "persist_assistant_result", persist)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.chat_completions(
                ChatRequest(question="q", conversation_id=uuid.uuid4()), http_request()
            )
        )
    assert exc.value.status_code == 401
    persist.assert_called_once()
    assert "secret://url" not in caplog.text
    pool.return_alive.assert_called_once_with(worker)


@pytest.mark.parametrize(
    "denial",
    [
        SourceSelectionError(409, "conversation_source_conflict"),
        router.ConversationAccessError("secret://url"),
    ],
)
def test_late_assistant_denial_is_not_retried(
    run_boundary: tuple[Mock, Mock, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    denial: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker, pool, prepare, auth = run_boundary
    worker.events.return_value = iter(
        [
            {"type": "result", "answer": {"response": "ok"}},
            {"type": "error", "message": "later"},
        ]
    )
    persist = Mock(side_effect=denial)
    monkeypatch.setattr(router, "persist_assistant_result", persist)
    monkeypatch.setattr(router, "_build_charts_event", lambda *args: None)
    slot = router._Slot(
        key="k",
        worker=worker,
        user_id="owner",
        conversation_id=uuid.uuid4(),
        analytics_id="a",
        question="q",
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
        source=SNAPSHOT,
    )
    router._pump(slot)
    persist.assert_called_once()
    assert persist.call_args.kwargs["connection_id"] == SOURCE.connection_id
    assert "secret://url" not in caplog.text
    pool.return_alive.assert_called_once_with(worker)


@pytest.mark.parametrize("denied", [False, True])
def test_chart_persistence_uses_accepted_source(
    run_boundary: tuple[Mock, Mock, Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    denied: bool,
) -> None:
    worker, pool, prepare, auth = run_boundary
    monkeypatch.setattr(router, "_non_reasoning_llm", None)
    monkeypatch.setattr(router, "_reasoning_llm", None)
    persist = Mock(
        side_effect=SourceSelectionError(404, "source_unavailable") if denied else None
    )
    monkeypatch.setattr(router, "persist_result_message", persist)
    slot = router._Slot(
        key="k",
        worker=worker,
        user_id="owner",
        conversation_id=uuid.uuid4(),
        analytics_id="a",
        question="q",
        client_alive=threading.Event(),
        cancelled=threading.Event(),
        released=threading.Event(),
        source=SNAPSHOT,
    )
    event = router._build_charts_event(slot, {"sql_response_from_db": '[{"n":1}]'})
    if denied:
        assert event is None
    else:
        assert event is not None and event["type"] == "charts"
    persist.assert_called_once()
    assert persist.call_args.kwargs["connection_id"] == SOURCE.connection_id
