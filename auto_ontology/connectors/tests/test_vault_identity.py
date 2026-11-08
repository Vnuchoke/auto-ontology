# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vault operations are exact-reference, fail-closed, and never bulk-delete."""

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.connectors import vault


def test_selected_ref_without_vault_fails_closed() -> None:
    with patch.object(vault, "is_vault_configured", return_value=False):
        with pytest.raises(ValueError, match="source_unavailable"):
            vault.read_secret("legacy/sales")


@pytest.mark.parametrize("reference", [None, "", "   "])
def test_delete_requires_exact_nonblank_ref(reference: str | None) -> None:
    with (
        patch.object(vault, "is_vault_configured", return_value=True),
        patch.object(vault, "get_client") as client,
    ):
        with pytest.raises(ValueError):
            vault.delete_secrets(reference)
    client.assert_not_called()


def test_exact_ref_roundtrip_and_safe_failure(caplog: pytest.LogCaptureFixture) -> None:
    client = MagicMock()
    client.secrets.kv.v1.read_secret.return_value = {
        "data": {"connection": '{"type": "postgresql", "password": "synthetic"}'}
    }
    with (
        patch.object(vault, "is_vault_configured", return_value=True),
        patch.object(vault, "get_client", return_value=client),
    ):
        payload = vault.read_secret("legacy/sales")
        vault.write_secret("legacy/sales", payload)
        vault.delete_secrets("legacy/sales")
        for method in ("read_secret", "create_or_update_secret", "delete_secret"):
            assert (
                getattr(client.secrets.kv.v1, method).call_args.kwargs["path"]
                == "legacy/sales"
            )
        client.secrets.kv.v1.read_secret.side_effect = RuntimeError(
            "synthetic-sensitive-url"
        )
        with pytest.raises(ValueError, match="source_unavailable") as error:
            vault.read_secret("legacy/sales")
    assert "synthetic-sensitive-url" not in str(error.value) + caplog.text
