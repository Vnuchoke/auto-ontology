# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reset transport never resolves a retired source by its reusable name."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.dal import reset as reset_dal
from auto_ontology.ingestion_service import ingest, router
from auto_ontology.server.ingestion import proxy
from auto_ontology.server.semantic_compilation import router as backend


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = FastAPI()
    app.state.semantic_scheduler = object()
    app.include_router(router.router)
    app.include_router(backend.router)
    forbidden = Mock(side_effect=AssertionError("source/credential lookup forbidden"))
    monkeypatch.setattr(router, "list_active_sources", forbidden)
    monkeypatch.setattr(ingest, "get_connector", forbidden)
    return TestClient(app)


def test_retired_delete_captures_id_before_label_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = []
    monkeypatch.setattr(
        ingest.threading,
        "Thread",
        lambda **kw: SimpleNamespace(start=lambda: jobs.append(kw["target"])),
    )
    sources = {"A": {"database_name": "sales", "display_label": "chosen"}}
    vectors = {"A": ["old"], "B": ["new"]}
    resolved = Mock(side_effect=sources.__getitem__)
    monkeypatch.setattr(ingest, "get_connector", resolved)

    def remove(cid: str) -> None:
        assert cid == "A"
        if cleanup.call_count == 1:
            raise RuntimeError("cleanup interrupted")
        vectors.pop(cid, None)

    cleanup = Mock(side_effect=remove)
    ordinary_reset = Mock()
    monkeypatch.setattr(reset_dal, "remove_connection_data", cleanup, raising=False)
    monkeypatch.setattr(ingest, "delete_all_data", ordinary_reset, raising=False)
    monkeypatch.setattr(reset_dal, "delete_all_data", ordinary_reset)
    ingest.trigger_delete_ingest("A")
    sources.pop("A")
    sources["B"] = {"database_name": "sales", "display_label": "chosen"}
    jobs[0]()
    assert vectors == {"A": ["old"], "B": ["new"]}
    jobs[0]()  # a retry keeps the captured identity even after the root is gone
    assert vectors == {"B": ["new"]}
    assert [call.args for call in cleanup.call_args_list] == [("A",), ("A",)]
    ordinary_reset.assert_not_called()
    resolved.assert_not_called()


def test_delete_unknown_retired_id_queues_without_active_lookup(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = Mock()
    monkeypatch.setattr(router, "trigger_delete_ingest", queued)
    response = client.post("/ingest/delete", params={"connection_id": "A"})
    assert response.status_code == 202
    assert response.json() == {"status": "accepted"}
    queued.assert_called_once_with("A")


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"connection_id": ""},
        {"connection_id": "  "},
        {"database_name": "sales"},
        {"connection_id": "A", "database_name": "sales"},
    ],
)
def test_delete_rejects_missing_blank_or_name_selectors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    params: dict,
) -> None:
    queued = Mock()
    monkeypatch.setattr(router, "trigger_delete_ingest", queued)
    assert client.post("/ingest/delete", params=params).status_code == 422
    queued.assert_not_called()


@pytest.mark.parametrize(
    "path, module, hook",
    [
        ("/semantic/reset", router, "trigger_reset_semantic"),
        ("/semantic-compilation/reset", backend, "trigger_semantic_reset"),
    ],
)
@pytest.mark.parametrize("cid", [None, "A", "a"])
def test_semantic_reset_forwards_opaque_or_explicit_global_scope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    module: object,
    hook: str,
    cid: str | None,
) -> None:
    queued = Mock()
    monkeypatch.setattr(module, hook, queued)
    params = {} if cid is None else {"connection_id": cid}
    assert client.post(path, params=params).status_code == 202
    assert queued.call_args.args[-1] == cid


@pytest.mark.parametrize("path", ["/semantic/reset", "/semantic-compilation/reset"])
@pytest.mark.parametrize(
    "params",
    [
        {"database_name": "sales"},
        {"connection_id": "A", "database_name": "sales"},
        {"connection_id": ""},
        {"connection_id": "  "},
    ],
)
def test_legacy_semantic_selector_cannot_become_global_wipe(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    params: dict,
) -> None:
    queued = Mock()
    monkeypatch.setattr(router, "trigger_reset_semantic", queued)
    monkeypatch.setattr(backend, "trigger_semantic_reset", queued)
    assert client.post(path, params=params).status_code == 422
    queued.assert_not_called()


@pytest.mark.parametrize("cid", [None, "A", "a"])
def test_semantic_reset_scope_waits_for_real_pass_then_deletes(
    monkeypatch: pytest.MonkeyPatch,
    cid: str | None,
) -> None:
    async def run() -> None:
        events = []
        finished = asyncio.Event()
        scheduler = SimpleNamespace(running=True)
        scheduler.pause = lambda: events.append("pause")
        scheduler.resume = lambda: events.append("resume")

        async def pass_work() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("drained")
                scheduler.running = False
                finished.set()

        task = asyncio.create_task(pass_work())
        await asyncio.sleep(0)

        def abort() -> None:
            events.append("abort")
            task.cancel()

        scheduler.abort = abort
        scheduler.start = lambda: events.append("start") or True

        async def delete(fn: object, selected: str | None) -> None:
            assert finished.is_set()
            assert fn is ingest.delete_semantic_layer
            assert selected == cid
            events.append("delete")

        monkeypatch.setattr(ingest.asyncio, "to_thread", delete)
        monkeypatch.setattr(ingest, "_ABORT_POLL_S", 0)
        await ingest.reset_semantic_layer_and_recompile(scheduler, connection_id=cid)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["pause", "abort", "drained", "delete", "resume", "start"]

    asyncio.run(run())


def test_cleanup_failure_log_does_not_expose_driver_secret(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        ingest.threading, "Thread", lambda **kw: SimpleNamespace(start=kw["target"])
    )
    failed = Mock(side_effect=RuntimeError("postgres://user:driver-secret@host/sales"))
    monkeypatch.setattr(reset_dal, "remove_connection_data", failed, raising=False)
    monkeypatch.setattr(ingest, "delete_all_data", failed, raising=False)
    ingest.trigger_delete_ingest("A")
    assert "A" in caplog.text
    assert "failed" in caplog.text
    assert "driver-secret" not in caplog.text
    assert failed.call_count == 1


@pytest.mark.parametrize("cid", [None, "A", "a"])
def test_proxy_sends_id_not_physical_name(
    monkeypatch: pytest.MonkeyPatch, cid: str | None
) -> None:
    post = Mock()
    monkeypatch.setattr(proxy.httpx, "post", post)
    proxy.trigger_semantic_reset(cid)
    assert post.call_args.kwargs["params"] == (
        {} if cid is None else {"connection_id": cid}
    )
    if cid is not None:
        proxy.trigger_reset(cid)
        assert post.call_args.kwargs["params"] == {"connection_id": cid}


@pytest.mark.parametrize("cid", [None, "A"])
def test_failed_semantic_reset_logs_only_scope_and_resumes_without_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cid: str | None,
) -> None:
    scheduler = SimpleNamespace(
        running=False,
        pause=Mock(),
        abort=Mock(),
        resume=Mock(),
        start=Mock(),
        trigger=Mock(),
    )
    failed = Mock(side_effect=RuntimeError("postgres://user:driver-secret@host/sales"))
    monkeypatch.setattr(ingest, "delete_semantic_layer", failed)
    asyncio.run(ingest.reset_semantic_layer_and_recompile(scheduler, cid))
    failed.assert_called_once_with(cid)
    scheduler.resume.assert_called_once()
    scheduler.start.assert_not_called()
    scheduler.trigger.assert_not_called()
    assert "failed" in caplog.text
    assert (cid or "<all>") in caplog.text
    assert "driver-secret" not in caplog.text


@pytest.mark.parametrize("hook", [proxy.trigger_reset, proxy.trigger_semantic_reset])
def test_failed_proxy_does_not_log_transport_credentials_or_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    hook: object,
) -> None:
    failed = Mock(side_effect=RuntimeError("http://user:driver-secret@host"))
    monkeypatch.setattr(proxy.httpx, "post", failed)
    monkeypatch.setenv("INGESTION_SERVICE_URL", "http://user:driver-secret@host")
    hook("A")
    assert failed.call_count == 1
    assert "A" in caplog.text
    assert "driver-secret" not in caplog.text
