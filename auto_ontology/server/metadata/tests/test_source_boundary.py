# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors.sources import SourceDescriptor, SourceSelectionError
from auto_ontology.retrieval.entity_coverage import main
from auto_ontology.server.metadata import router, service

SOURCE = SourceDescriptor("a", "Sales A", "sales", "postgres", "managed")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


@pytest.mark.parametrize(
    "status,code",
    [
        (503, "source_unavailable"),
        (422, "source_required"),
        (404, "source_not_found"),
        (409, "ambiguous_source"),
        (422, "invalid_source_selector"),
        (503, "source_not_ready"),
    ],
)
def test_selection_errors_precede_models_and_connectors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    code: str,
) -> None:
    resolver = Mock(side_effect=SourceSelectionError(status, code))
    connectors = Mock(side_effect=AssertionError("must not open connectors"))
    invoke = Mock(side_effect=AssertionError("must not run graph"))
    monkeypatch.setattr(service, "resolve_query_source", resolver)
    monkeypatch.setattr(service, "get_connectors_for_subject_token", connectors)
    monkeypatch.setattr(service, "get_coverage_response", invoke)
    monkeypatch.setattr(service, "coverage_llm_client", None)
    response = client.post("/question-entity-coverage", json={"question": "orders"})
    assert response.status_code == status
    assert response.json()["detail"] == {"code": code}
    connectors.assert_not_called()
    invoke.assert_not_called()


@pytest.mark.parametrize("selector", ["connection_id", "connection_label", "target_db"])
def test_selectors_are_strict_strings(client: TestClient, selector: str) -> None:
    assert (
        client.post(
            "/question-entity-coverage", json={"question": "orders", selector: 42}
        ).status_code
        == 422
    )


def test_direct_entrypoint_preflights_before_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "resolve_query_source",
        Mock(side_effect=SourceSelectionError(503, "source_not_ready")),
    )
    monkeypatch.setattr(main, "llm_client", None)
    with pytest.raises(SourceSelectionError, match="source_not_ready"):
        main.get_coverage_response({"question": "orders"})


def test_direct_state_uses_one_cid_and_safe_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "resolve_query_source", Mock(return_value=SOURCE))
    monkeypatch.setattr(main, "llm_client", object())
    rules = Mock(return_value=[])
    monkeypatch.setattr(main, "fetch_custom_analyses", rules)
    invoke = Mock(
        return_value={
            "path_state": {
                "final_response": {
                    "coverage": 1,
                    "candidates": [],
                    "source": {"secret": "bad"},
                }
            }
        }
    )
    monkeypatch.setattr(main, "app", SimpleNamespace(invoke=invoke))
    selected, other = (
        SimpleNamespace(connection_id="a"),
        SimpleNamespace(connection_id="b"),
    )
    response = main.get_coverage_response(
        {
            "question": "orders",
            "connection_id": "a",
            "data_retriever": object(),
            "connectors": [selected, other],
        }
    )
    state = invoke.call_args.args[0]
    assert state["connectors"] == [selected]
    assert state["path_state"]["connection_id"] == "a"
    assert state["path_state"]["target_db"] == "sales"
    assert state["connection_id"] == "a"
    assert response["source"] == {
        "connection_id": "a",
        "display_label": "Sales A",
        "database_name": "sales",
    }
    rules.assert_called_once_with(connection_id="a")


def test_api_forwards_selected_sso_token_without_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.connectors.databricks_oauth import DatabricksOAuthError

    monkeypatch.setattr(service, "resolve_query_source", Mock(return_value=SOURCE))
    monkeypatch.setattr(service, "coverage_llm_client", object())
    connectors = Mock(
        side_effect=DatabricksOAuthError("Selected source SSO token exchange failed")
    )
    monkeypatch.setattr(service, "get_connectors_for_subject_token", connectors)
    response = client.post(
        "/question-entity-coverage",
        headers={"Authorization": "Bearer caller-token"},
        json={"question": "orders", "connection_id": "a"},
    )
    assert response.status_code == 401
    connectors.assert_called_once_with("caller-token", connection_id="a")


def test_selected_connector_failure_is_safe_503_without_neighbor_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "resolve_query_source", Mock(return_value=SOURCE))
    monkeypatch.setattr(service, "coverage_llm_client", object())
    connectors = Mock(side_effect=RuntimeError("secret://offline"))
    monkeypatch.setattr(service, "get_connectors_for_subject_token", connectors)
    response = client.post(
        "/question-entity-coverage", json={"question": "orders", "connection_id": "a"}
    )
    assert response.status_code == 503
    assert response.json()["detail"] == {"code": "source_unavailable"}
    connectors.assert_called_once_with(None, connection_id="a")
