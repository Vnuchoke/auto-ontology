# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from auto_ontology.semantic import embed
from auto_ontology.semantic.embed import _format_sample_values


def test_format_sample_values_handles_legacy_json_string_and_native_list() -> None:
    # Legacy Column nodes still store sample_values as a JSON string.
    assert _format_sample_values('["a", "b"]') == " Sample values: a, b."
    # Current writers store a native list.
    assert _format_sample_values(["a", "b"]) == " Sample values: a, b."


def test_format_sample_values_filters_long_values_and_handles_empty() -> None:
    long_value = "x" * 31
    assert _format_sample_values([long_value, "ok"]) == " Sample values: ok."
    assert _format_sample_values(None) == ""
    assert _format_sample_values([]) == ""
    assert _format_sample_values(["a", None, "b"]) == " Sample values: a, b."
    assert _format_sample_values('["a", null, "b"]') == " Sample values: a, b."


@pytest.mark.parametrize("owner", [None, "b"])
def test_direct_embed_rejects_unscoped_or_foreign_content_before_model(
    monkeypatch: pytest.MonkeyPatch, owner: str | None
) -> None:
    batch = Mock()
    monkeypatch.setattr(embed, "batch_embed", batch)
    embedder = embed.SemanticEmbedder("sales", "a", Mock(), Mock())
    with pytest.raises(ValueError, match="connection_id"):
        embedder.embed_term(
            {"name": "Order", "connection_id": "a"},
            [{"name": "Status", "connection_id": owner, "table_id": "t"}],
        )
    with pytest.raises(ValueError, match="connection_id"):
        embedder.embed_term({"name": "Order", "connection_id": owner}, [])
    batch.assert_not_called()


def test_physical_name_is_metadata_not_embedding_text_and_json_limit_stays() -> None:
    term = {"id": "shared", "name": "Order", "connection_id": "a"}
    attr = {"id": "status", "name": "Status", "table_id": "t", "connection_id": "a"}
    first = embed._build_rows("sales", term, [attr], connection_id="a")
    second = embed._build_rows("renamed_physical", term, [attr], connection_id="a")
    assert [r["text"] for r in first] == [r["text"] for r in second]
    assert first[0]["metadata"]["database_name"] == "sales"
    sample = "key [e.g. '" + "x" * 30 + "']"
    assert _format_sample_values([sample]) == ""
    assert sample in _format_sample_values([sample], "jsonb")
