# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure helpers in auto_ontology.semantic.embed."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from auto_ontology.semantic import embed
from auto_ontology.semantic.embed import _build_rows, _format_sample_values


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


def _attr_text(**attr: object) -> str:
    base = {
        "id": "a1",
        "name": "Total",
        "term_name": "Order",
        "source_column": "t",
        "connection_id": "a",
        "table_id": "ta",
    }
    (row,) = _build_rows("shop", {}, [{**base, **attr}], connection_id="a")
    return row["text"]


def test_the_columns_description_is_embedded_not_the_attributes() -> None:
    text = _attr_text(
        description="The long attribute description",
        column_description="Order total",
        sample_values="[10, 20]",
    )

    assert (
        text
        == "ColumnAttribute: Total of Term Order. Order total Sample values: 10, 20."
    )


def test_the_family_is_named_after_the_term() -> None:
    (row,) = _build_rows(
        "shop",
        {"synonyms": ["Purchase"]},
        [
            {
                "id": "a1",
                "name": "Total",
                "term_name": "Order",
                "column_description": "Order total",
                "family_prefix": "NEW_",
                "connection_id": "a",
                "table_id": "ta",
            }
        ],
        connection_id="a",
    )

    assert row["text"] == (
        "ColumnAttribute: Total of Term Order (Purchase), family NEW_. Order total"
    )


def test_an_undocumented_column_embeds_its_name_and_samples() -> None:
    assert _attr_text(sample_values='["a"]') == (
        "ColumnAttribute: Total of Term Order.  Sample values: a."
    )


def test_embed_term_rows_validates_source_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Mock()
    monkeypatch.setattr(embed, "batch_embed", batch)
    embedder = embed.SemanticEmbedder("sales", "a", Mock(), Mock())
    with pytest.raises(ValueError, match="connection_id"):
        embedder.embed_term_rows({"name": "B_ONLY", "connection_id": "b"}, [])
    batch.assert_not_called()


def test_same_name_sources_embed_only_their_projected_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    seen = []

    def batch(rows, _params):
        seen.extend(rows)
        return pd.DataFrame(
            [
                {**row, "metadata": {**row["metadata"], "embedding": [1.0]}}
                for row in rows
            ]
        )

    monkeypatch.setattr(embed, "batch_embed", batch)
    for cid in ("a", "b", "a"):
        embedder = embed.SemanticEmbedder("sales", cid, Mock(), Mock())
        rows = embedder.embed_term_rows(
            {"id": "shared", "name": "Order", "connection_id": cid},
            [
                {
                    "id": cid,
                    "name": "Status",
                    "table_id": cid,
                    "connection_id": cid,
                    "column_description": f"{cid.upper()}_ONLY",
                    "family_prefix": cid,
                }
            ],
        )
        assert all(row["metadata"]["connection_id"] == cid for row in rows)
        text = repr(rows)
        assert f"{cid.upper()}_ONLY" in text
        assert ("B_ONLY" if cid == "a" else "A_ONLY") not in text
        embedder.vdb.assert_not_called()
    assert len(seen) == 6
