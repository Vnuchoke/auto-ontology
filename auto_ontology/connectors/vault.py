# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HashiCorp Vault access for UI-managed database connection secrets.

Callers supply an exact credential reference, never a physical-name fallback.
Missing configuration or an unavailable selected secret fails closed.

This module performs no import-time side effects (it does not load ``.env``);
entrypoints are responsible for loading the environment before use.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
import hvac

logger = logging.getLogger(__name__)

REQUIRED_VAULT_ENV_VARS = (
    "VAULT_ADDR",
    "VAULT_NAMESPACE",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
)


def is_vault_configured() -> bool:
    """Return True if the user provided all required Vault env vars.

    Warn on partial configuration (some but not all set), which is almost
    always a misconfiguration that would otherwise silently disable Vault.
    """
    missing = [var for var in REQUIRED_VAULT_ENV_VARS if not os.environ.get(var)]
    if missing and len(missing) < len(REQUIRED_VAULT_ENV_VARS):
        logger.warning(
            "Vault is partially configured; ignoring Vault and treating "
            "connections as unencrypted. Missing env vars: %s",
            ", ".join(missing),
        )
    return not missing


def get_client() -> hvac.Client:
    addr = os.environ["VAULT_ADDR"]
    auth_mount = os.environ.get("VAULT_AUTH_MOUNT", "approle/nvdcs/dc1")
    try:
        client = hvac.Client(
            url=addr,
            namespace=os.environ["VAULT_NAMESPACE"],
        )
        client.auth.approle.login(
            role_id=os.environ["VAULT_ROLE_ID"],
            secret_id=os.environ["VAULT_SECRET_ID"],
            mount_point=auth_mount,
        )
    except Exception:
        raise ValueError("source_unavailable: Vault authentication failed") from None
    return client


def _require_reference(reference: str) -> None:
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("invalid_credential_reference")
    if not is_vault_configured():
        raise ValueError("source_unavailable: Vault is not configured")


def write_secret(database_name: str, secret: dict[str, Any]) -> None:
    _require_reference(database_name)
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    try:
        client = get_client()
        client.secrets.kv.v1.create_or_update_secret(
            path=database_name,
            secret={"connection": json.dumps(secret)},
            mount_point=mount_point,
        )
    except Exception:
        raise ValueError("source_unavailable: Vault write failed") from None


def read_secret(database_name: str) -> dict[str, Any]:
    _require_reference(database_name)
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    try:
        client = get_client()
        response = client.secrets.kv.v1.read_secret(
            path=database_name,
            mount_point=mount_point,
        )
        payload = json.loads(response["data"]["connection"])
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except Exception:
        raise ValueError("source_unavailable: Vault read failed") from None


def delete_secrets(database_name: str) -> None:
    """Delete exactly one reference; blank or omitted paths never mean delete all."""
    _require_reference(database_name)
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    try:
        get_client().secrets.kv.v1.delete_secret(
            path=database_name, mount_point=mount_point
        )
    except Exception:
        raise ValueError("source_unavailable: Vault delete failed") from None
