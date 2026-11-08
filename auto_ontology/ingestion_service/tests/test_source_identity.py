# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion boundaries preserve opaque source IDs, not physical names."""

import asyncio
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors.sources import SourceDescriptor


@pytest.fixture
def sources() -> list[SourceDescriptor]:
    return [
        SourceDescriptor(cid, cid, "sales", "postgres", "env") for cid in ("A", "a")
    ]


def test_both_schedulers_discover_the_same_metadata_only_set(
    monkeypatch: pytest.MonkeyPatch, sources: list[SourceDescriptor]
) -> None:
    from auto_ontology.ingestion_service import (
        connections,
        data_scheduler,
        semantic_scheduler,
    )
    from auto_ontology.connectors import registry
    from auto_ontology.dal import connections as connection_dal

    metadata = Mock(return_value=sources)
    credentials = Mock(side_effect=AssertionError("credential enumeration forbidden"))
    monkeypatch.setattr(connection_dal, "list_connections", credentials)
    monkeypatch.setattr(registry, "create_connector", credentials)
    monkeypatch.setattr(connections, "list_active_sources", metadata, raising=False)
    connectors = {
        s.connection_id: SimpleNamespace(
            connection_id=s.connection_id, database_name="sales"
        )
        for s in sources
    }
    opened = Mock(side_effect=connectors.__getitem__)
    ingested = Mock()
    compiled = Mock(return_value=1)
    monkeypatch.setattr(data_scheduler, "invalidate_connectors_cache", lambda: None)
    monkeypatch.setattr(data_scheduler, "get_connector", opened, raising=False)
    monkeypatch.setattr(
        data_scheduler,
        "get_connectors",
        Mock(side_effect=AssertionError("credential enumeration forbidden")),
        raising=False,
    )
    monkeypatch.setattr(data_scheduler, "run_ingest", ingested)
    monkeypatch.setattr(data_scheduler, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(
        semantic_scheduler, "is_semantic_compilation_enabled", lambda: True
    )
    monkeypatch.setattr(semantic_scheduler, "run_semantic_compilation", compiled)
    monkeypatch.setattr(semantic_scheduler, "record_run_start", lambda: None)
    monkeypatch.setattr(semantic_scheduler, "record_run_finish", lambda *args: None)

    propagated: list[str] = []

    async def propagate(label: str, *, connection_id: str) -> None:
        propagated.append(connection_id)

    async def noop(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(semantic_scheduler, "run_pii_propagation_in_thread", propagate)
    monkeypatch.setattr(semantic_scheduler, "replay_rules_in_thread", noop)

    asyncio.run(data_scheduler.DataScheduler()._run_once())
    asyncio.run(semantic_scheduler.SemanticScheduler()._run_once())

    assert metadata.call_count == 2
    assert [call.args[0] for call in opened.call_args_list] == ["A", "a"]
    assert [call.args[0].connection_id for call in ingested.call_args_list] == [
        "A",
        "a",
    ]
    assert [call.args[0] for call in compiled.call_args_list] == ["A", "a"]
    assert propagated == ["A", "a"]
    credentials.assert_not_called()


def test_connector_failure_is_counted_without_aborting_other_sources(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from auto_ontology.ingestion_service import data_scheduler as mod

    monkeypatch.setattr(
        mod, "resolve_connection_ids", lambda: ["B", "A"], raising=False
    )
    monkeypatch.setattr(mod, "invalidate_connectors_cache", lambda: None)

    def connector(cid: str) -> SimpleNamespace:
        if cid == "B":
            raise ValueError("unavailable")
        return SimpleNamespace(connection_id=cid)

    monkeypatch.setattr(mod, "get_connector", connector, raising=False)
    monkeypatch.setattr(
        mod,
        "get_connectors",
        Mock(side_effect=AssertionError("credential enumeration forbidden")),
        raising=False,
    )
    ingested = Mock()
    monkeypatch.setattr(mod, "run_ingest", ingested)
    monkeypatch.setattr(mod, "is_semantic_compilation_enabled", lambda: True)
    asyncio.run(mod.DataScheduler()._run_once())
    assert ingested.call_args.args[0].connection_id == "A"
    assert "1 of 2 connection(s) succeeded, 1 failed" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"connection_id": ""},
        {"connection_id": "  "},
        {"connection_id": 7},
        {"connection_id": "unknown"},
        {"database": "sales", "password": "secret"},
    ],
)
def test_invalid_ingest_never_queues_or_opens_a_driver(
    monkeypatch: pytest.MonkeyPatch,
    sources: list[SourceDescriptor],
    payload: dict[str, Any],
) -> None:
    from auto_ontology.ingestion_service import router as mod
    from auto_ontology.connectors import registry
    from auto_ontology import utils

    opened = Mock(side_effect=AssertionError("driver boundary reached"))
    model = Mock(side_effect=AssertionError("model boundary reached"))
    monkeypatch.setattr(registry, "get_connector", opened)
    monkeypatch.setattr(registry, "create_connector", opened)
    monkeypatch.setattr(utils, "get_embed_params", model)
    monkeypatch.setattr(mod, "list_active_sources", lambda: sources, raising=False)
    queued = Mock()
    monkeypatch.setattr(mod, "trigger_ingest", queued)
    app = FastAPI()
    app.include_router(mod.router)
    response = TestClient(app).post("/ingest", json=payload)
    assert response.status_code == (
        404 if payload.get("connection_id") == "unknown" else 422
    )
    queued.assert_not_called()
    opened.assert_not_called()
    model.assert_not_called()


def test_valid_ingest_queues_only_selected_id(
    monkeypatch: pytest.MonkeyPatch, sources: list[SourceDescriptor]
) -> None:
    from auto_ontology.ingestion_service import router as mod

    monkeypatch.setattr(mod, "list_active_sources", lambda: sources, raising=False)
    queued = Mock()
    monkeypatch.setattr(mod, "trigger_ingest", queued)
    app = FastAPI()
    app.include_router(mod.router)
    assert (
        TestClient(app).post("/ingest", json={"connection_id": "a"}).status_code == 202
    )
    queued.assert_called_once_with("a")


@pytest.mark.parametrize("cid", [None, "", " ", 7])
def test_ingest_rejects_untrusted_connector_before_model_or_writes(
    monkeypatch: pytest.MonkeyPatch, cid: Any
) -> None:
    from auto_ontology.ingestion_service import ingest as mod

    params = Mock(side_effect=AssertionError("model boundary reached"))
    monkeypatch.setattr(mod, "get_embed_params", params)
    with pytest.raises(ValueError, match="connection_id"):
        mod.run_ingest(SimpleNamespace(connection_id=cid, database_name="sales"))
    params.assert_not_called()


def test_ingest_resets_only_selected_vectors_on_first_successful_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.ingestion_service import ingest as mod

    snapshot = {"A": ["old-a"], "a": ["old-b"]}
    params = {}
    monkeypatch.setattr(mod, "get_embed_params", lambda: params)
    monkeypatch.setattr(mod, "ingest_catalog", lambda c: ([], []))
    monkeypatch.setattr(mod, "is_pii_detection_enabled", lambda: False)
    rows = Mock(return_value=lambda pair: pair)
    monkeypatch.setattr(mod, "CatalogEmbeddingRowsOp", rows)

    def chunks(*args: Any, **kwargs: Any) -> Iterator[list[str]]:
        yield []
        assert snapshot == {"A": ["old-a"], "a": ["old-b"]}
        yield ["new-a"]

    monkeypatch.setattr(mod, "batch_embed_chunks", chunks)
    monkeypatch.setattr(mod, "to_vdb_records", lambda chunk: chunk)

    def vdb(*, connection_id: str, database_name: str, reset: bool) -> SimpleNamespace:
        assert (connection_id, database_name, reset) == ("A", "sales", True)
        snapshot[connection_id] = []

        def write(records: list[str]) -> int:
            snapshot[connection_id].extend(records)
            return len(records)

        return SimpleNamespace(run=write)

    monkeypatch.setattr(mod, "get_data_vdb", vdb)
    mod.run_ingest(SimpleNamespace(connection_id="A", database_name="sales"))
    rows.assert_called_once_with(database_name="sales", connection_id="A")
    assert snapshot == {"A": ["new-a"], "a": ["old-b"]}


@pytest.mark.parametrize("embedding_fails", [False, True])
def test_empty_or_failed_embedding_preserves_live_vectors(
    monkeypatch: pytest.MonkeyPatch, embedding_fails: bool
) -> None:
    from auto_ontology.ingestion_service import ingest as mod

    monkeypatch.setattr(mod, "get_embed_params", lambda: {})
    monkeypatch.setattr(mod, "ingest_catalog", lambda c: ([], []))
    monkeypatch.setattr(mod, "is_pii_detection_enabled", lambda: False)
    monkeypatch.setattr(mod, "CatalogEmbeddingRowsOp", lambda **kw: lambda pair: pair)

    def chunks(*args: Any, **kwargs: Any) -> Iterator[list[str]]:
        yield []
        if embedding_fails:
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(mod, "batch_embed_chunks", chunks)
    monkeypatch.setattr(mod, "to_vdb_records", lambda chunk: chunk)
    reset = Mock(side_effect=AssertionError("live vectors must survive"))
    monkeypatch.setattr(mod, "get_data_vdb", reset)
    connector = SimpleNamespace(connection_id="A", database_name="sales")
    if embedding_fails:
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            mod.run_ingest(connector)
    else:
        mod.run_ingest(connector)
    reset.assert_not_called()


def test_trigger_opens_only_selected_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_ontology.ingestion_service import ingest as mod

    selected = SimpleNamespace(connection_id="a", database_name="sales")
    opened = Mock(return_value=selected)
    monkeypatch.setattr(mod, "get_connector", opened, raising=False)
    monkeypatch.setattr(mod, "invalidate_connectors_cache", lambda: None)
    ingested = Mock()
    monkeypatch.setattr(mod, "run_ingest", ingested)
    monkeypatch.setattr(mod, "replay_rules", lambda label: None)

    class InlineThread:
        def __init__(self, *, target: Callable[[], None], **kwargs: Any) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(mod.threading, "Thread", InlineThread)
    mod.trigger_ingest("a")
    opened.assert_called_once_with("a")
    ingested.assert_called_once_with(selected)


def test_proxy_sends_only_source_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_ontology.server.ingestion import proxy

    post = Mock()
    monkeypatch.setattr(proxy.httpx, "post", post)
    proxy.trigger_ingest("a")
    assert post.call_args.kwargs["json"] == {"connection_id": "a"}


def test_local_cli_bootstraps_before_selecting_first_registered_source(
    monkeypatch: pytest.MonkeyPatch, sources: list[SourceDescriptor]
) -> None:
    from dev_tools import local_ingest as mod

    order = []
    monkeypatch.setattr(mod, "require_current_schema", lambda: order.append("schema"))
    monkeypatch.setattr(mod, "bootstrap_sources", lambda: order.append("bootstrap"))

    def discover() -> list[SourceDescriptor]:
        order.append("metadata")
        return sources

    monkeypatch.setattr(mod, "list_active_sources", discover)
    selected = SimpleNamespace(connection_id="A", database_name="sales")
    opened = Mock(return_value=selected)
    ingested = Mock()
    monkeypatch.setattr(mod, "get_connector", opened)
    monkeypatch.setattr(mod, "ingest_connector", ingested)
    mod.main()
    assert order == ["schema", "bootstrap", "metadata"]
    opened.assert_called_once_with("A")
    ingested.assert_called_once_with(selected)


def test_local_cli_without_active_sources_does_not_open_connectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dev_tools import local_ingest as mod

    monkeypatch.setattr(mod, "require_current_schema", lambda: None)
    monkeypatch.setattr(mod, "bootstrap_sources", lambda: None)
    monkeypatch.setattr(mod, "list_active_sources", lambda: [])
    opened = Mock()
    monkeypatch.setattr(mod, "get_connector", opened)
    with pytest.raises(EnvironmentError, match="No active sources"):
        mod.main()
    opened.assert_not_called()
