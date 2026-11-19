# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Withholding PII values from the catalog and the semantic VDB."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from auto_ontology.semantic.embed import SAMPLE_VALUES_MARKER
from auto_ontology.semantic.pii_values import (
    PiiWithholdingResult,
    withhold_pii_column_values,
)

_MODULE = "auto_ontology.semantic.pii_values"

_TERMS = [
    {
        "id": "term-user",
        "name": "User",
        "synonyms": ["Account"],
        "connection_id": "source-a",
    },
    {"id": "term-order", "name": "Order", "synonyms": [], "connection_id": "source-a"},
]
_ATTRS = [
    {
        "id": "a-email",
        "source_column": "email",
        "term_name": "User",
        "sample_values": ["stale-canary"],
    },
    {"id": "a-phone", "source_column": "phone", "term_name": "User"},
    {"id": "a-status", "source_column": "status", "term_name": "Order"},
]
for _attr in _ATTRS:
    _attr.update(connection_id="source-a", table_id="t1")
_PII_ATTRIBUTES = [
    {
        "attribute_id": "a-email",
        "table_id": "t1",
        "database_name": "db",
        "connection_id": "source-a",
    },
    {
        "attribute_id": "a-phone",
        "table_id": "t1",
        "database_name": "db",
        "connection_id": "source-a",
    },
]


def _vdb(stale: list[str]) -> MagicMock:
    vdb = MagicMock()
    vdb.ids_with_content_containing.return_value = stale
    vdb.delete_by_id.return_value = 1
    return vdb


def _withhold(
    *,
    cleared: int = 0,
    attributes: list[dict] = _PII_ATTRIBUTES,
    vdbs: dict[str, MagicMock],
    embedder: MagicMock | None,
) -> PiiWithholdingResult:
    with (
        patch(f"{_MODULE}.clear_values_of_pii_columns", return_value=cleared),
        patch(f"{_MODULE}.fetch_attributes_of_pii_columns", return_value=attributes),
        patch(
            f"{_MODULE}.get_semantic_vdb",
            side_effect=lambda *, database_name, connection_id: vdbs[connection_id],
        ),
        patch(
            f"{_MODULE}.fetch_terms_and_attributes_for_table",
            return_value=(_TERMS, _ATTRS),
        ),
        patch(f"{_MODULE}.build_semantic_embedder", return_value=embedder),
    ):
        return withhold_pii_column_values()


def test_only_vectors_still_holding_values_are_replaced() -> None:
    vdb = _vdb(stale=["a-email"])
    embedder = MagicMock()
    embedder.embed_term.return_value = 1

    result = _withhold(cleared=2, vdbs={"source-a": vdb}, embedder=embedder)

    assert result == PiiWithholdingResult(
        columns_cleared=2, vectors_deleted=1, vectors_reembedded=1
    )
    vdb.ids_with_content_containing.assert_called_once_with(
        ["a-email", "a-phone"], SAMPLE_VALUES_MARKER, connection_id="source-a"
    )
    vdb.delete_by_id.assert_called_once_with("a-email", connection_id="source-a")
    embedder.embed_term.assert_called_once_with(
        {"synonyms": ["Account"]},
        [{**_ATTRS[0], "sample_values": None, "is_unique": None}],
    )


def test_a_stale_vector_is_found_after_the_catalog_was_cleared() -> None:
    """The retry: an earlier pass cleared the profile but not the vector."""
    vdb = _vdb(stale=["a-email"])

    result = _withhold(cleared=0, vdbs={"source-a": vdb}, embedder=MagicMock())

    assert result.columns_cleared == 0
    vdb.delete_by_id.assert_called_once_with("a-email", connection_id="source-a")


def test_stale_vectors_are_deleted_without_an_embedder() -> None:
    vdb = _vdb(stale=["a-email", "a-phone"])

    result = _withhold(vdbs={"source-a": vdb}, embedder=None)

    assert result.vectors_deleted == 2
    assert result.vectors_reembedded == 0
    assert vdb.delete_by_id.call_count == 2


def test_clean_vectors_touch_nothing() -> None:
    vdb = _vdb(stale=[])
    embedder = MagicMock()

    result = _withhold(vdbs={"source-a": vdb}, embedder=embedder)

    assert result == PiiWithholdingResult()
    vdb.delete_by_id.assert_not_called()
    embedder.embed_term.assert_not_called()


def test_one_unreachable_vdb_does_not_stop_same_name_source() -> None:
    broken = _vdb(stale=[])
    broken.ids_with_content_containing.side_effect = RuntimeError("vdb down")
    healthy = _vdb(stale=["a-other"])
    attributes = [
        _PII_ATTRIBUTES[0],
        {
            "attribute_id": "a-other",
            "table_id": "t2",
            "database_name": "db",
            "connection_id": "source-b",
        },
    ]

    result = _withhold(
        attributes=attributes,
        vdbs={"source-a": broken, "source-b": healthy},
        embedder=None,
    )

    assert result.vectors_deleted == 1
    healthy.delete_by_id.assert_called_once_with("a-other", connection_id="source-b")


def test_a_failed_re_embed_leaves_the_vector_deleted() -> None:
    vdb = _vdb(stale=["a-email"])
    embedder = MagicMock()
    embedder.embed_term.side_effect = RuntimeError("embed down")

    result = _withhold(vdbs={"source-a": vdb}, embedder=embedder)

    assert result.vectors_deleted == 1
    assert result.vectors_reembedded == 0
