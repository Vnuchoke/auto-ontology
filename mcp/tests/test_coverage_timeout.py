# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage gets a bounded read budget without changing other concurrent tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from auto_ontology_mcp import server
from auto_ontology_mcp.config import load_settings


def _build_server(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[..., Any],
) -> tuple[Any, httpx.AsyncClient]:
    monkeypatch.setattr(server, "auto_ontology_access_token", lambda: "caller-token")
    original = httpx.AsyncClient
    # Keep the production client factory, auth and hooks; replace only network I/O.
    with monkeypatch.context() as patch:
        patch.setattr(
            httpx,
            "AsyncClient",
            lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
        )
        return server.build_server(load_settings())


@pytest.mark.parametrize("api_path", ["", "/tenant"])
@pytest.mark.parametrize("budget", [None, "240.5", "30"])
@pytest.mark.parametrize("status", [200, 503])
def test_concurrent_tools_keep_isolated_budgets_and_identity(
    monkeypatch: pytest.MonkeyPatch, api_path: str, budget: str | None, status: int
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_API_URL", f"http://ontology.test{api_path}")
    monkeypatch.setenv("AUTO_ONTOLOGY_MCP_TIMEOUT_S", "30")
    monkeypatch.setenv("AUTO_ONTOLOGY_MCP_CHAT_TIMEOUT_S", "900")
    monkeypatch.delenv("AUTO_ONTOLOGY_OPENAPI_SPEC", raising=False)
    monkeypatch.delenv("AUTO_ONTOLOGY_MCP_COVERAGE_TIMEOUT_S", raising=False)
    if budget is not None:
        monkeypatch.setenv("AUTO_ONTOLOGY_MCP_COVERAGE_TIMEOUT_S", budget)
    expected_read = float(budget) if budget is not None else 180.0
    captured: dict[str, httpx.Request] = {}

    async def run() -> None:
        arrived = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path.removeprefix(api_path)
            captured[path] = request
            assert client.timeout.as_dict() == default_timeout
            if len(captured) == 3:
                arrived.set()
            await arrived.wait()
            if path == "/api/question-entity-coverage":
                # MockTransport does not enforce timeouts; simulate a 120s read
                # deterministically, without waiting or weakening production limits.
                if request.extensions["timeout"]["read"] < 120:
                    raise httpx.ReadTimeout(
                        "simulated 120s coverage read", request=request
                    )
                return httpx.Response(status, json={"data": {"coverage": 1}})
            if path == "/api/chat/completions":
                event = {"type": "result", "answer": {"response": "Covered."}}
                return httpx.Response(
                    200,
                    text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            assert path == "/api/connections/sources"
            return httpx.Response(200, json={"data": [], "count": 0})

        mcp, client = _build_server(monkeypatch, handler)
        default_timeout = {key: 30.0 for key in ("connect", "read", "write", "pool")}
        assert client.timeout.as_dict() == default_timeout
        try:
            async with Client(mcp) as session:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        session.call_tool(
                            "check_answerable",
                            {"question": "sales?", "connection_id": "selected-cid"},
                        ),
                        session.call_tool("list_sources", {}),
                        session.call_tool("ask_question", {"question": "sales?"}),
                        return_exceptions=True,
                    ),
                    timeout=10,
                )
            if expected_read < 120 or status != 200:
                assert isinstance(results[0], ToolError)
            else:
                assert results[0].structured_content == {"data": {"coverage": 1}}
            assert not isinstance(results[1], BaseException)
            assert not isinstance(results[2], BaseException)
            assert client.timeout.as_dict() == default_timeout
        finally:
            await client.aclose()

    asyncio.run(run())
    coverage = captured["/api/question-entity-coverage"]
    assert coverage.method == "POST"
    assert json.loads(coverage.content) == {
        "question": "sales?",
        "connection_id": "selected-cid",
    }
    assert coverage.extensions["timeout"] == {
        "connect": 30.0,
        "read": expected_read,
        "write": 30.0,
        "pool": 30.0,
    }
    assert captured["/api/connections/sources"].extensions["timeout"] == {
        key: 30.0 for key in ("connect", "read", "write", "pool")
    }
    assert captured["/api/chat/completions"].extensions["timeout"] == {
        key: 900.0 for key in ("connect", "read", "write", "pool")
    }
    assert all(
        request.headers["authorization"] == "Bearer caller-token"
        for request in captured.values()
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/question-entity-coverage"),
        ("POST", "/api/question-entity-coverage/extra"),
        ("POST", "/unrelated/api/question-entity-coverage"),
    ],
)
def test_coverage_budget_does_not_match_other_routes(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_API_URL", "http://ontology.test")
    monkeypatch.setenv("AUTO_ONTOLOGY_MCP_TIMEOUT_S", "30")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    _, client = _build_server(monkeypatch, handler)

    async def run() -> None:
        async with client:
            await client.request(method, path)

    asyncio.run(run())
    assert captured[0].extensions["timeout"] == {
        key: 30.0 for key in ("connect", "read", "write", "pool")
    }
