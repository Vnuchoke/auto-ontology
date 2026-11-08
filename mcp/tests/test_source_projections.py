# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated tools carry source context through the published OpenAPI contract."""

import asyncio
import json

import httpx
import pytest
from fastmcp import Client

from auto_ontology_mcp import server
from auto_ontology_mcp.config import DEFAULT_SPEC_PATH, Settings


@pytest.mark.parametrize(
    "selector", [None, "connection_id", "connection_label", "target_db"]
)
@pytest.mark.parametrize(
    ("tool", "arguments", "path"),
    [
        ("search_terms", {}, "/api/terms"),
        ("get_term", {"term_id": "shared"}, "/api/terms/shared"),
        (
            "get_term_columns",
            {"term_id": "shared"},
            "/api/terms/shared/column-attributes",
        ),
        (
            "get_term_sql_attributes",
            {"term_id": "shared"},
            "/api/terms/shared/sql-attributes",
        ),
        ("get_sql_attribute", {"attr_id": "metric"}, "/api/sql-attributes/metric"),
        (
            "describe_table",
            {"table_id": "orders"},
            "/api/exploration/tables/orders/details",
        ),
        ("check_answerable", {"question": "sales?"}, "/api/question-entity-coverage"),
    ],
)
def test_generated_source_selectors_are_serialized_without_auto_selection(
    monkeypatch: pytest.MonkeyPatch,
    selector: str | None,
    tool: str,
    arguments: dict[str, str],
    path: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payloads = {
            "search_terms": {
                "terms": [],
                "total": 0,
                "column_attribute_counts": [],
                "sql_attribute_counts": [],
                "related_counts": [],
            },
            "get_term": {"data": {"id": "shared"}},
            "get_term_columns": {"data": [], "count": 0, "total": 0},
            "get_term_sql_attributes": {"data": [], "count": 0, "total": 0},
            "get_sql_attribute": {"data": {"id": "metric"}},
            "describe_table": {"data": {"queries": [], "terms": [], "terms_total": 0}},
            "check_answerable": {"data": {"coverage": 0}},
        }
        return httpx.Response(200, json=payloads[tool])

    settings = Settings(
        api_url="http://ontology.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=30,
        chat_timeout_s=900,
        public_url="http://127.0.0.1:3003",
    )
    client = httpx.AsyncClient(
        base_url=settings.api_url, transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr(server, "build_client", lambda _: client)
    mcp, _ = server.build_server(settings)
    selected = {selector: "East"} if selector else {}

    async def run() -> None:
        try:
            async with Client(mcp) as session:
                tools = {item.name: item for item in await session.list_tools()}
                assert set(tools[tool].inputSchema["properties"]) >= {
                    "connection_id",
                    "connection_label",
                    "target_db",
                }
                await session.call_tool(tool, {**arguments, **selected})
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == path
    if tool == "check_answerable":
        assert requests[0].method == "POST"
        assert json.loads(requests[0].content) == {**arguments, **selected}
    else:
        assert requests[0].method == "GET"
        assert dict(requests[0].url.params) == selected
