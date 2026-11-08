# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Curated reads resolve optional metadata selectors before any semantic read."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors import sources
from auto_ontology.server.terms import router, service
from auto_ontology.server.sql_attributes import router as sql_router
from auto_ontology.server.exploration import router as exploration_router


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        lambda: [
            sources.SourceDescriptor(cid, label, "sales", "postgres", "managed")
            for cid, label in [("a", "local"), ("b", "production")]
        ],
    )
    monkeypatch.setattr(
        sources.connections,
        "normalize_source_labels",
        lambda labels: [s.strip().lower() for s in labels],
    )
    app = FastAPI()
    app.include_router(router.router, prefix="/api")
    app.include_router(sql_router.router, prefix="/api")
    app.include_router(exploration_router.router, prefix="/api")
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "selector", [{"connection_id": "a"}, {"connection_label": "local"}]
)
def test_list_scopes_every_read(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, selector: dict[str, str]
) -> None:
    calls = []
    for module, name, result in [
        (service.terms_dal, "fetch_all_terms", []),
        (service.terms_dal, "count_terms", 5),
        (service.terms_dal, "fetch_column_attribute_counts", []),
        (service.sql_attr_dal, "fetch_sql_attribute_counts", []),
        (service.terms_dal, "fetch_related_terms_counts", []),
    ]:
        spy = Mock(return_value=result)
        monkeypatch.setattr(module, name, spy)
        calls.append(spy)
    response = client.get("/api/terms", params={**selector, "limit": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 5
    assert all(spy.call_args.kwargs.get("connection_id") == "a" for spy in calls)


@pytest.mark.parametrize(
    "path",
    [
        "/terms",
        "/terms/t",
        "/terms/t/column-attributes",
        "/terms/t/sql-attributes",
        "/sql-attributes/x",
        "/exploration/tables/t/details",
    ],
)
@pytest.mark.parametrize(
    "selector,status,code",
    [
        ({"connection_id": "missing"}, 404, "source_unavailable"),
        ({"connection_id": ""}, 422, "invalid_source_selector"),
        ({"connection_id": "a", "target_db": "sales"}, 422, "invalid_source_selector"),
        ({"target_db": "sales"}, 409, "ambiguous_source"),
    ],
)
def test_invalid_source_precedes_dal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    selector: dict[str, str],
    status: int,
    code: str,
) -> None:
    spy = Mock(side_effect=AssertionError("semantic DAL must not run"))
    for module, name in [
        (service, "list_terms_page"),
        (router.terms_dal, "get_full_term_by_id"),
        (router.terms_dal, "fetch_column_attributes_by_term_id"),
        (router.sql_attr_dal, "fetch_sql_attributes_by_term_id"),
        (sql_router.dal, "get_full_sql_attribute_by_id"),
        (exploration_router.service, "fetch_table_exploration_details"),
    ]:
        monkeypatch.setattr(module, name, spy)
    response = client.get("/api" + path, params=selector)
    assert response.status_code == status
    assert response.json() == {"detail": {"code": code}}
    spy.assert_not_called()


@pytest.mark.parametrize(
    "path,name,module",
    [
        (
            "/terms/t/column-attributes",
            "fetch_column_attributes_by_term_id",
            router.terms_dal,
        ),
        (
            "/terms/t/sql-attributes",
            "fetch_sql_attributes_by_term_id",
            router.sql_attr_dal,
        ),
    ],
)
def test_attribute_page_and_total_share_source(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    name: str,
    module: object,
) -> None:
    fetch = Mock(return_value=[])
    count = Mock(return_value=7)
    monkeypatch.setattr(module, name, fetch)
    monkeypatch.setattr(module, name.replace("fetch_", "count_"), count)
    response = client.get(
        "/api" + path, params={"connection_id": "a", "skip": 2, "limit": 1}
    )
    assert response.status_code == 200
    assert response.json() == {"data": [], "count": 0, "total": 7}
    assert (
        fetch.call_args.kwargs["connection_id"]
        == count.call_args.kwargs["connection_id"]
        == "a"
    )


@pytest.mark.parametrize("active_count", [0, 1, 2])
def test_unselected_glossary_never_resolves_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, active_count: int
) -> None:
    metadata = Mock(
        return_value=[
            sources.SourceDescriptor(str(i), str(i), "sales", "postgres", "managed")
            for i in range(active_count)
        ]
    )
    monkeypatch.setattr(sources, "list_active_sources", metadata)
    resolver = Mock(side_effect=AssertionError("global read must not select a source"))
    monkeypatch.setattr(sources, "resolve_source", resolver)
    spy = Mock(
        return_value={
            "terms": [],
            "total": 0,
            "column_attribute_counts": [],
            "sql_attribute_counts": [],
            "related_counts": [],
        }
    )
    monkeypatch.setattr(service, "list_terms_page", spy)
    assert client.get("/api/terms").status_code == 200
    assert spy.call_args.kwargs.get("connection_id") is None
    resolver.assert_not_called()
    metadata.assert_not_called()


def test_detail_and_related_share_resolved_legacy_source(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        lambda: [
            sources.SourceDescriptor("a", "local", "sales", "postgres", "managed")
        ],
    )
    detail = Mock(return_value={"id": "t", "name": "Common"})
    related = Mock(return_value=[])
    monkeypatch.setattr(router.terms_dal, "get_full_term_by_id", detail)
    monkeypatch.setattr(router.terms_dal, "fetch_related_terms", related)
    response = client.get("/api/terms/t", params={"target_db": "sales"})
    assert response.status_code == 200
    assert (
        detail.call_args.kwargs["connection_id"]
        == related.call_args.kwargs["connection_id"]
        == "a"
    )
