# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.server.connections import router, service


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router.router, prefix="/api")
    return TestClient(app)


def _row(identity: str = "local") -> dict:
    return {
        "connection_id": identity,
        "display_label": identity.title(),
        "database_name": "sales",
        "connection": {
            "type": "postgresql",
            "database": "sales",
            "host": "synthetic.invalid",
            "password": "PRIVATE",
            "access_token_override": "PRIVATE",
            "credential_ref": "PRIVATE",
        },
    }


def test_same_named_connections_keep_ids_and_redact_envelopes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row("local"), _row("remote")]
    before = deepcopy(rows)
    monkeypatch.setattr(router, "_list_connections", lambda: rows)
    result = client.get("/api/connections")
    assert result.status_code == 200
    payload = result.json()
    assert [item["connection_id"] for item in payload["data"]] == ["local", "remote"]
    assert [item["database_name"] for item in payload["data"]] == ["sales", "sales"]
    assert "PRIVATE" not in result.text
    assert rows == before


def test_create_sends_label_separately_and_returns_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    create = Mock(return_value=_row())
    monkeypatch.setattr(service, "create_connection", create)
    result = client.post(
        "/api/connections",
        json={
            "connection": {"type": "postgresql", "database": "sales"},
            "display_label": "Local",
        },
    )
    assert result.status_code == 201
    create.assert_called_once_with(
        connection={"type": "postgresql", "database": "sales"}, display_label="Local"
    )
    assert result.json()["data"]["connection_id"] == "local"
    assert "PRIVATE" not in result.text


def test_by_id_label_only_update_does_not_require_driver_settings(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    update = Mock(return_value={**_row(), "connection": {}})
    monkeypatch.setattr(service, "update_connection", update)
    result = client.put(
        "/api/connections/by-id/local", json={"display_label": "Renamed"}
    )
    assert result.status_code == 200
    update.assert_called_once_with(
        connection_id="local", connection=None, display_label="Renamed"
    )
    assert result.json()["data"]["connection_id"] == "local"


def test_id_test_and_sso_reach_only_the_selected_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = Mock(return_value=["public"])
    toggle = Mock(return_value={**_row(), "sso_federation": True})
    monkeypatch.setattr(service, "test_connection", probe)
    monkeypatch.setattr(service, "set_sso_federation", toggle)
    result = client.post(
        "/api/connections/by-id/local/test",
        json={"connection": {"database": "sales", "password": ""}},
    )
    assert result.status_code == 200
    probe.assert_called_once_with(
        {"database": "sales", "password": ""}, replacing_connection_id="local"
    )
    result = client.patch(
        "/api/connections/by-id/local/sso-federation", json={"enabled": True}
    )
    assert result.status_code == 200
    toggle.assert_called_once_with(connection_id="local", enabled=True)
    assert result.json()["data"]["connection_id"] == "local"
    assert "PRIVATE" not in result.text and "connection" not in result.json()["data"]


def test_id_delete_returns_captured_identity(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row()
    row.pop("connection")
    delete = Mock(return_value=row)
    monkeypatch.setattr(service, "delete_connection", delete)
    result = client.delete("/api/connections/by-id/local")
    assert result.status_code == 200
    delete.assert_called_once_with(connection_id="local")
    assert result.json()["data"] == row


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("put", "/api/connections/sales", {"connection": {"database": "sales"}}),
        ("delete", "/api/connections/sales", None),
        ("patch", "/api/connections/sales/sso-federation", {"enabled": True}),
    ],
)
def test_ambiguous_legacy_adapter_returns_conflict_before_mutation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    monkeypatch.setattr(
        service,
        "list_source_metadata",
        lambda: [
            {"id": i, "name": "sales", "configuration_origin": "managed"}
            for i in ("a", "b")
        ],
    )
    write = Mock(side_effect=AssertionError("ambiguous source must not mutate"))
    monkeypatch.setattr(service, "insert_connection", write)
    monkeypatch.setattr(service, "clear_connection", write)
    monkeypatch.setattr(service, "load_source_connection", write)
    result = client.request(method, path, json=payload)
    assert result.status_code == 409
    assert "ambiguous_source" in result.text
    write.assert_not_called()


@pytest.mark.parametrize(
    "error,status",
    [
        (service.LabelConflictError("label_conflict"), 409),
        (service.SourceUnavailableError("source_unavailable"), 503),
        (RuntimeError("PRIVATE"), 422),
    ],
)
def test_controlled_status_and_no_raw_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    monkeypatch.setattr(service, "create_connection", Mock(side_effect=error))
    result = client.post("/api/connections", json={"connection": {"database": "sales"}})
    assert result.status_code == status
    assert "PRIVATE" not in result.text


def test_empty_update_and_conflicting_test_selectors_are_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    update = Mock()
    probe = Mock()
    monkeypatch.setattr(service, "update_connection", update)
    monkeypatch.setattr(service, "test_connection", probe)
    assert client.put("/api/connections/by-id/local", json={}).status_code == 422
    assert (
        client.post(
            "/api/connections/test",
            json={
                "connection": {"database": "sales"},
                "replacing": "sales",
                "replacing_connection_id": "local",
            },
        ).status_code
        == 422
    )
    update.assert_not_called()
    probe.assert_not_called()
