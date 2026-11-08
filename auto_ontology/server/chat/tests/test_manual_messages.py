# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual append only crosses the identity and transactional DAL boundary."""

import uuid
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors.sources import SourceSelectionError
from auto_ontology.server.chat import router


@pytest.fixture
def manual(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Mock, uuid.UUID]:
    app = FastAPI()
    app.include_router(router.router)
    cid = uuid.uuid4()
    append = Mock(
        return_value={
            "id": "message",
            "conversation_id": str(cid),
            "role": "user",
            "content": "q",
            "sql_code": None,
            "sql_response": None,
            "created_at": "2026-11-08T00:00:00Z",
        }
    )
    monkeypatch.setattr(router, "append_message", append)
    monkeypatch.setattr(router, "get_pool", Mock(side_effect=AssertionError("worker")))
    monkeypatch.setattr(
        router, "_subject_token", Mock(side_effect=AssertionError("SSO"))
    )
    monkeypatch.setattr(
        router, "get_llm_client", Mock(side_effect=AssertionError("model"))
    )
    monkeypatch.setattr(router, "resolve_internal_user", lambda req, required: "owner")
    return TestClient(app), append, cid


def test_manual_forwards_selectors_and_preserves_created_shape(
    manual: tuple[TestClient, Mock, uuid.UUID],
) -> None:
    client, append, cid = manual
    body = {"role": "user", "content": "q", "connection_label": "East"}
    response = client.post(f"/chat/conversations/{cid}/messages", json=body)
    assert response.status_code == 201
    assert response.json() == append.return_value
    append.assert_called_once_with(
        conversation_id=cid,
        user_id="owner",
        role="user",
        content="q",
        sql_code=None,
        sql_response=None,
        connection_id=None,
        connection_label="East",
        target_db=None,
    )


@pytest.mark.parametrize(
    "status,code",
    [
        (409, "conversation_source_conflict"),
        (409, "legacy_conversation_read_only"),
        (404, "source_unavailable"),
        (422, "invalid_source_selector"),
    ],
)
def test_manual_source_errors_survive(
    manual: tuple[TestClient, Mock, uuid.UUID], status: int, code: str
) -> None:
    client, append, cid = manual
    append.side_effect = SourceSelectionError(status, code)
    response = client.post(
        f"/chat/conversations/{cid}/messages", json={"role": "assistant"}
    )
    assert response.status_code == status
    assert response.json() == {"detail": code}


@pytest.mark.parametrize(
    "error",
    [router.ConversationAccessError("secret"), RuntimeError("postgres://secret")],
)
def test_manual_owner_and_storage_errors_are_sanitized(
    manual: tuple[TestClient, Mock, uuid.UUID], error: Exception
) -> None:
    client, append, cid = manual
    append.side_effect = error
    response = client.post(f"/chat/conversations/{cid}/messages", json={"role": "user"})
    assert response.status_code == (
        404 if isinstance(error, router.ConversationAccessError) else 503
    )
    assert response.json() == {
        "detail": "Conversation not found"
        if isinstance(error, router.ConversationAccessError)
        else "Conversation storage unavailable"
    }
    append.assert_called_once()
    assert "secret" not in response.text


@pytest.mark.parametrize("field", ["connection_id", "connection_label", "target_db"])
def test_manual_nonstring_selector_is_422(
    manual: tuple[TestClient, Mock, uuid.UUID], field: str
) -> None:
    client, append, cid = manual
    response = client.post(
        f"/chat/conversations/{cid}/messages", json={"role": "user", field: 12}
    )
    assert response.status_code == 422
    append.assert_not_called()


def test_manual_requires_private_identity(
    manual: tuple[TestClient, Mock, uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.server.identity import resolve_internal_user

    client, append, cid = manual
    monkeypatch.setattr(router, "resolve_internal_user", resolve_internal_user)
    response = client.post(f"/chat/conversations/{cid}/messages", json={"role": "user"})
    assert response.status_code == 401
    append.assert_not_called()
