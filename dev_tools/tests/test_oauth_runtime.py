# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in discovery checks against the local Compose stack; no login or DB writes."""

import json
import os
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTO_ONTOLOGY_TEST_RUNTIME") != "1",
    reason="Requires the running local Compose stack",
)

ISSUER = "http://localhost:18300/api/auth"


def _metadata(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        assert response.status == 200
        return json.load(response)


def test_mcp_advertises_the_better_auth_issuer() -> None:
    resource = _metadata(
        "http://localhost:18303/.well-known/oauth-protected-resource/mcp"
    )

    assert resource["authorization_servers"] == [ISSUER]
    assert resource["resource"] == "http://localhost:18303/mcp"


def test_rfc8414_discovery_returns_the_exact_advertised_issuer() -> None:
    metadata = _metadata(
        "http://localhost:18300/.well-known/oauth-authorization-server/api/auth"
    )

    assert metadata["issuer"] == ISSUER
    assert metadata["authorization_endpoint"] == f"{ISSUER}/oauth2/authorize"
    assert metadata["token_endpoint"] == f"{ISSUER}/oauth2/token"

    legacy = _metadata("http://localhost:18300/.well-known/oauth-authorization-server")
    for key in (
        "issuer",
        "authorization_endpoint",
        "token_endpoint",
        "registration_endpoint",
    ):
        assert metadata[key] == legacy[key]
