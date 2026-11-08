# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import psycopg
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.server.chat import router
from auto_ontology.server.chat.tests.test_conversation_source_binding import (
    A,
    B,
    _seed,
    _snapshot,
    identity_database as identity_database,
)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router.router, prefix="/api")
    return TestClient(app, headers={"x-auto-ontology-user-id": "owner"})


def test_manual_http_uses_real_binding_transaction_without_execution(
    identity_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = uuid4()
    _seed(conversation)
    execution = Mock(side_effect=AssertionError("manual append must not execute"))
    monkeypatch.setattr(router, "get_pool", execution)
    monkeypatch.setattr(router, "_subject_token", execution)
    path = f"/api/chat/conversations/{conversation}/messages"
    with _client() as client:
        first = client.post(
            path, json={"role": "user", "content": "one", "connection_label": "Sales A"}
        )
        assert first.status_code == 201
        assert set(first.json()) == {
            "id",
            "conversation_id",
            "role",
            "content",
            "sql_code",
            "sql_response",
            "created_at",
        }
        assert (
            client.post(path, json={"role": "assistant", "content": "two"}).status_code
            == 201
        )
        before = _snapshot()
        conflict = client.post(path, json={"role": "user", "connection_id": B})
        assert conflict.status_code == 409
        assert conflict.json() == {"detail": "source_conflict"}
        assert _snapshot() == before
    conversations, messages, analytics = _snapshot()
    assert conversations[0]["connection_id"] == A
    assert len(messages) == 2 and analytics == []
    execution.assert_not_called()


def test_manual_http_owner_legacy_and_inactive_rejections_are_atomic(
    identity_database: None,
) -> None:
    legacy, bound = uuid4(), uuid4()
    _seed(legacy, role="assistant")
    _seed(bound, source=A)
    with _client() as client:
        before = _snapshot()
        missing = client.post(
            f"/api/chat/conversations/{uuid4()}/messages",
            json={"role": "user", "connection_id": A},
        )
        foreign = client.post(
            f"/api/chat/conversations/{bound}/messages",
            headers={"x-auto-ontology-user-id": "foreign"},
            json={"role": "user", "connection_id": A},
        )
        assert missing.status_code == foreign.status_code == 404
        assert missing.json() == foreign.json() == {"detail": "Conversation not found"}
        refused = client.post(
            f"/api/chat/conversations/{legacy}/messages",
            json={"role": "user", "connection_id": A},
        )
        assert refused.status_code == 409
        assert refused.json() == {"detail": "legacy_conversation_read_only"}
        assert _snapshot() == before
        with psycopg.connect(get_postgres_connection_string()) as conn:
            conn.execute(
                "UPDATE catalog_database SET configuration_origin = NULL WHERE id = %s",
                (A,),
            )
        inactive = client.post(
            f"/api/chat/conversations/{bound}/messages", json={"role": "user"}
        )
        assert inactive.status_code == 404
        assert inactive.json() == {"detail": "source_unavailable"}
        assert _snapshot() == before


@pytest.mark.parametrize("sso_failure", [False, True])
def test_chat_http_inherits_real_binding_and_persists_accepted_cid(
    identity_database: None, monkeypatch: pytest.MonkeyPatch, sso_failure: bool
) -> None:
    conversation = uuid4()
    _seed(conversation, source=A)
    worker = Mock()
    worker.events.return_value = iter(
        [{"type": "result", "answer": {"response": "answer", "sql_code": "SELECT 1"}}]
    )
    pool = Mock()
    pool.acquire.return_value = worker
    monkeypatch.setattr(router, "get_pool", lambda: pool)
    monkeypatch.setattr(router, "_active_slots", {})
    token = Mock(
        return_value=None,
        side_effect=HTTPException(401, "SSO required") if sso_failure else None,
    )
    monkeypatch.setattr(router, "_subject_token", token)
    with _client() as client:
        response = client.post(
            "/api/chat/completions",
            json={"question": "question", "conversation_id": str(conversation)},
        )
    token.assert_called_once()
    assert token.call_args.args[1] == A
    if sso_failure:
        assert response.status_code == 401
        worker.submit.assert_not_called()
    else:
        assert response.status_code == 200
        assert f'"connection_id": "{A}"' in response.text
        assert worker.submit.call_args.kwargs["connection_id"] == A
    conversations, messages, analytics = _snapshot()
    assert conversations[0]["connection_id"] == A
    assert [
        message["role"]
        for message in sorted(messages, key=lambda row: (row["created_at"], row["id"]))
    ] == ["user", "assistant"]
    assert len(analytics) == 1 and analytics[0]["response_timestamp"] is not None
    if sso_failure:
        assert any(
            message["content"].startswith("Agent failed:") for message in messages
        )
    assert router._active_slots == {}
