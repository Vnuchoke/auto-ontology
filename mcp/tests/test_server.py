# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The assembled server publishes the curated surface and nothing else."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.server.http import set_http_request
from starlette.requests import Request

from auto_ontology_mcp import server
from auto_ontology_mcp.config import (
    DEFAULT_SPEC_PATH,
    ConfigError,
    Settings,
    load_settings,
)
from auto_ontology_mcp.server import (
    ICON_PATH,
    INSTRUCTIONS,
    CallerAuth,
    build_client,
    build_server,
    load_icons,
    load_spec,
)
from auto_ontology_mcp.tools import CURATED
from auto_ontology_mcp import get_version


def _settings(spec_path: Path = DEFAULT_SPEC_PATH) -> Settings:
    return Settings(
        api_url="http://auto_ontology.test",
        spec_path=spec_path,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=900.0,
        public_url="http://127.0.0.1:3003",
    )


def _tools() -> list[Any]:
    mcp, client = build_server(_settings())

    async def run() -> list[Any]:
        try:
            async with Client(mcp) as session:
                return await session.list_tools()
        finally:
            await client.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("auth_url", "expected_issuer"),
    [
        ("https://ontology.example", "https://ontology.example/api/auth"),
        ("https://ontology.example/", "https://ontology.example/api/auth"),
        ("https://ontology.example///", "https://ontology.example/api/auth"),
        (None, "http://frontend:3000/api/auth"),
    ],
)
def test_metadata_advertises_auth_issuer(
    monkeypatch: pytest.MonkeyPatch, auth_url: str | None, expected_issuer: str
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_API_URL", "http://frontend:3000")
    if auth_url is None:
        monkeypatch.delenv("AUTO_ONTOLOGY_AUTH_URL", raising=False)
    else:
        monkeypatch.setenv("AUTO_ONTOLOGY_AUTH_URL", auth_url)
    monkeypatch.setenv("AUTO_ONTOLOGY_MCP_PUBLIC_URL", "https://mcp.example")
    monkeypatch.delenv("AUTO_ONTOLOGY_OPENAPI_SPEC", raising=False)
    mcp, client = build_server(load_settings())

    async def run() -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mcp.http_app()),
                base_url="https://mcp.example",
            ) as session:
                response = await session.get(
                    "/.well-known/oauth-protected-resource/mcp"
                )
                assert response.status_code == 200
                return response.json()
        finally:
            await client.aclose()

    metadata = asyncio.run(run())

    assert metadata["authorization_servers"] == [expected_issuer]
    assert metadata["resource"] == "https://mcp.example/mcp"


def test_public_auth_url_does_not_redirect_tool_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_API_URL", "http://frontend:3000")
    monkeypatch.setenv("AUTO_ONTOLOGY_AUTH_URL", "https://ontology.example")
    monkeypatch.setenv("AUTO_ONTOLOGY_MCP_PUBLIC_URL", "https://mcp.example")
    monkeypatch.delenv("AUTO_ONTOLOGY_OPENAPI_SPEC", raising=False)
    monkeypatch.setattr(server, "auto_ontology_access_token", lambda: "test-token")
    requests: list[httpx.Request] = []

    async def handle(
        _transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "terms": [],
                "total": 0,
                "column_attribute_counts": [],
                "sql_attribute_counts": [],
                "related_counts": [],
            },
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    mcp, client = build_server(load_settings())

    async def run() -> None:
        try:
            async with Client(mcp) as session:
                await session.call_tool("search_terms", {})
        finally:
            await client.aclose()

    asyncio.run(run())

    assert len(requests) == 1
    assert requests[0].url.host == "frontend"
    assert requests[0].url.port == 3000
    assert requests[0].url.path == "/api/terms"
    assert requests[0].headers["Authorization"] == "Bearer test-token"


def test_publishes_exactly_the_curated_tools_plus_the_handwritten_ones() -> None:
    names = {tool.name for tool in _tools()}

    assert names == {spec.name for spec in CURATED} | {
        "ask_question",
        "check_readiness",
    }
    assert names == {
        "list_sources",
        "search_terms",
        "get_term",
        "get_term_columns",
        "get_term_sql_attributes",
        "get_sql_attribute",
        "describe_table",
        "check_answerable",
        "ask_question",
        "check_readiness",
    }


def test_every_tool_carries_a_description() -> None:
    # An undescribed tool is close to unusable: selection is driven almost
    # entirely by this text.
    undescribed = [
        tool.name for tool in _tools() if not (tool.description or "").strip()
    ]

    assert undescribed == []


def test_generated_descriptions_are_replaced_end_to_end() -> None:
    # test_tools.py checks the override function in isolation; this checks it
    # was actually handed to FastMCP and ran over the real spec.
    leaked = [tool.name for tool in _tools() if "Api." in (tool.description or "")]

    assert leaked == []


def test_ask_question_reports_a_structured_answer() -> None:
    ask = next(tool for tool in _tools() if tool.name == "ask_question")

    assert ask.outputSchema is not None
    assert set(ask.outputSchema["properties"]) >= {"answer", "sql", "rows", "source"}
    assert set(ask.inputSchema["properties"]) == {
        "question",
        "conversation_id",
        "evidence",
        "connection_id",
        "connection_label",
    }


@pytest.mark.parametrize("status", [200, 403])
def test_generated_list_sources_calls_discovery_as_the_signed_in_caller(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    requests: list[httpx.Request] = []
    sources = [
        {
            "connection_id": "local-id",
            "display_label": "sales-local",
            "database_name": "sales",
            "dialect": "postgresql",
        },
        {
            "connection_id": "prod-id",
            "display_label": "sales-prod",
            "database_name": "sales",
            "dialect": "postgresql",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            json={"data": sources, "count": 2}
            if status == 200
            else {"detail": "Missing chat:use permission"},
        )

    client = httpx.AsyncClient(
        base_url=_settings().api_url,
        auth=CallerAuth(),
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(server, "build_client", lambda settings: client)
    monkeypatch.setattr(server, "auto_ontology_access_token", lambda: "chat-only-token")
    mcp, owned_client = build_server(_settings())

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return await session.call_tool("list_sources", {})
        finally:
            await owned_client.aclose()

    if status == 403:
        with pytest.raises(ToolError, match="403"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result.structured_content == {"data": sources, "count": 2}
        assert all(
            set(source)
            == {
                "connection_id",
                "display_label",
                "database_name",
                "dialect",
            }
            for source in result.structured_content["data"]
        )
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/connections/sources"
    assert requests[0].url.query == b""
    assert requests[0].headers["authorization"] == "Bearer chat-only-token"


def test_handshake_advertises_the_auto_ontology_version() -> None:
    # Not FastMCP's, which is what gets reported if the version is left unset
    # and reads as a Auto Ontology version in a client's server list.
    mcp, client = build_server(_settings())

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return session.initialize_result.serverInfo
        finally:
            await client.aclose()

    info = asyncio.run(run())

    assert info.name == "auto-ontology"
    assert info.version == get_version()


def test_handshake_advertises_the_icon() -> None:
    mcp, client = build_server(_settings())

    async def run() -> Any:
        try:
            async with Client(mcp) as session:
                return session.initialize_result.serverInfo
        finally:
            await client.aclose()

    icons = asyncio.run(run()).icons or []

    assert len(icons) == 1
    assert icons[0].mimeType == "image/svg+xml"
    # Inline, so a client can draw it without reaching the network or Auto Ontology.
    assert icons[0].src.startswith("data:image/svg+xml;base64,")


def test_the_icon_decodes_back_to_the_svg() -> None:
    src = load_icons()[0].src
    payload = base64.b64decode(src.removeprefix("data:image/svg+xml;base64,"))

    assert payload == ICON_PATH.read_bytes()
    assert payload.startswith(b"<svg")


def test_a_missing_icon_does_not_stop_the_server(monkeypatch: Any) -> None:
    # Cosmetic, so it must never be the reason a working server fails to start.
    monkeypatch.setattr(server, "ICON_PATH", Path("/nonexistent/nvidia-mark.svg"))

    assert load_icons() == []


def test_instructions_point_at_the_primary_tool() -> None:
    # Sequencing guidance lives here rather than in any one tool description,
    # so a host that shows only this still knows where to start.
    assert "ask_question" in INSTRUCTIONS
    assert "check_answerable" in INSTRUCTIONS
    assert "check_readiness" in INSTRUCTIONS
    assert "list_sources" in INSTRUCTIONS
    assert "connection_id" in INSTRUCTIONS
    assert "`check_answerable` with the selected source" in INSTRUCTIONS


def _sent_headers(
    auth: CallerAuth, incoming: dict[str, str] | None = None
) -> httpx.Headers:
    """Run one request through *auth* and return the headers it went out with.

    The incoming request is faked with FastMCP's own HTTP request context, so
    this exercises the same lookup a live HTTP transport would.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    async def run() -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            auth=auth,
            base_url="http://auto_ontology.test",
        )
        try:
            await client.get("/api/terms")
        finally:
            await client.aclose()

    if incoming is None:
        asyncio.run(run())
    else:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(k.lower().encode(), v.encode()) for k, v in incoming.items()],
        }
        with set_http_request(Request(scope)):
            asyncio.run(run())

    return captured[0].headers


def test_no_credential_is_attached_to_the_client_itself() -> None:
    """A shared client must not carry one caller's identity for all callers."""
    client = build_client(_settings())
    try:
        assert "x-api-key" not in client.headers
    finally:
        asyncio.run(client.aclose())


def test_the_signed_in_callers_token_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto Ontology issued it, so it goes upstream untouched — no exchange, no unwrapping."""
    monkeypatch.setattr(
        server, "auto_ontology_access_token", lambda: "auto-ontology-issued"
    )

    headers = _sent_headers(CallerAuth())

    assert headers["authorization"] == "Bearer auto-ontology-issued"


def test_a_smuggled_api_key_cannot_override_the_signed_in_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastMCP copies the caller's headers onto the outbound request, and Auto Ontology
    resolves x-api-key ahead of a bearer token. Left in place, a caller could
    sign in as one user and then call Auto Ontology as the owner of some other token.
    """
    monkeypatch.setattr(
        server, "auto_ontology_access_token", lambda: "auto-ontology-issued"
    )

    headers = _sent_headers(CallerAuth(), {"x-api-key": "auto_ontology_someone_else"})

    assert headers["authorization"] == "Bearer auto-ontology-issued"
    assert "x-api-key" not in headers


def test_a_smuggled_bearer_token_cannot_override_it_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server, "auto_ontology_access_token", lambda: "auto-ontology-issued"
    )

    headers = _sent_headers(CallerAuth(), {"authorization": "Bearer someone-else"})

    assert headers["authorization"] == "Bearer auto-ontology-issued"


def test_a_request_with_no_signed_in_session_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "auto_ontology_access_token", lambda: None)

    with pytest.raises(ToolError, match="Sign in again"):
        _sent_headers(CallerAuth())


def test_rejects_a_spec_that_is_not_json(tmp_path: Path) -> None:
    bad = tmp_path / "spec.json"
    bad.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid JSON"):
        load_spec(_settings(bad))


def test_rejects_a_spec_with_no_paths(tmp_path: Path) -> None:
    empty = tmp_path / "spec.json"
    empty.write_text(json.dumps({"openapi": "3.1.0"}), encoding="utf-8")

    with pytest.raises(ConfigError, match="no paths"):
        load_spec(_settings(empty))


def test_refuses_to_start_when_a_curated_endpoint_disappeared(tmp_path: Path) -> None:
    # Starting anyway would produce a healthy-looking server missing a tool.
    spec = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    del spec["paths"]["/api/terms"]
    trimmed = tmp_path / "spec.json"
    trimmed.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(ConfigError, match="GET /api/terms"):
        load_spec(_settings(trimmed))
