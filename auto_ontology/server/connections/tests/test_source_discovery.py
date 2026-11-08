# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors import sources
from auto_ontology.server.connections import router


def test_discovery_metadata_only_safe_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_ontology.connectors import registry

    for name in ("get_connectors", "get_connectors_for_subject_token"):
        monkeypatch.setattr(registry, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.setattr(
        router, "_list_connections", Mock(side_effect=AssertionError("settings read"))
    )
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        lambda: [
            sources.SourceDescriptor("A", "East", "sales", "postgres", "managed"),
            sources.SourceDescriptor("B", "West", "sales", "postgres", "env"),
        ],
    )
    monkeypatch.setattr(
        sources, "bootstrap_sources", Mock(side_effect=AssertionError("bootstrap"))
    )
    app = FastAPI()
    app.include_router(router.router)
    response = TestClient(app).get("/connections/sources")
    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "connection_id": "A",
                "display_label": "East",
                "database_name": "sales",
                "dialect": "postgres",
            },
            {
                "connection_id": "B",
                "display_label": "West",
                "database_name": "sales",
                "dialect": "postgres",
            },
        ],
        "count": 2,
    }


def test_discovery_metadata_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sources, "list_active_sources", Mock(side_effect=RuntimeError("secret-url"))
    )
    app = FastAPI()
    app.include_router(router.router)
    response = TestClient(app).get("/connections/sources")
    assert response.status_code == 503
    assert response.json() == {"detail": "source_not_ready"}
