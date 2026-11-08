# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local Compose stack keeps private APIs private and OAuth client-reachable."""

from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


def test_explicit_container_names_are_project_prefixed_and_unique() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    names = [
        service["container_name"]
        for service in services.values()
        if "container_name" in service
    ]

    assert not [
        name
        for name in names
        if name != "auto-ontology" and not name.startswith("auto-ontology-")
    ]
    assert len(names) == len(set(names))


def test_only_public_services_publish_loopback_ports() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]

    assert {
        name: service["ports"]
        for name, service in services.items()
        if "ports" in service
    } == {
        "postgres": ["127.0.0.1:${POSTGRES_PORT}:5432"],
        "pgadmin": ["127.0.0.1:18350:80"],
        "auto-ontology-frontend": ["127.0.0.1:18300:3000"],
        "auto-ontology-mcp": ["127.0.0.1:18303:3003"],
    }


def test_mcp_uses_internal_api_and_public_oauth_without_shared_credentials() -> None:
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    mcp = services["auto-ontology-mcp"]

    assert mcp["build"] == {"context": "./mcp"}
    assert "env_file" not in mcp
    assert mcp["environment"] == {
        "AUTO_ONTOLOGY_API_URL": "http://auto-ontology-frontend:3000",
        "AUTO_ONTOLOGY_AUTH_URL": "${APP_URL}",
        "AUTO_ONTOLOGY_MCP_PUBLIC_URL": "${AUTO_ONTOLOGY_MCP_PUBLIC_URL}",
        "AUTO_ONTOLOGY_MCP_HOST": "0.0.0.0",
        "AUTO_ONTOLOGY_MCP_PORT": "3003",
    }
    assert mcp["depends_on"] == {
        "auto-ontology-frontend": {"condition": "service_healthy"}
    }


def test_example_matches_local_compose_ports_and_valid_external_rerank_url() -> None:
    example = dotenv_values(ROOT / ".env.example", interpolate=False)

    assert example["POSTGRES_PORT"] == "18332"
    assert example["APP_URL"] == "http://localhost:18300"
    assert example["AUTO_ONTOLOGY_MCP_PUBLIC_URL"] == "http://localhost:18303"
    assert example["RERANK_ENDPOINT"] == "http://host.docker.internal:18304/v1/rerank"
