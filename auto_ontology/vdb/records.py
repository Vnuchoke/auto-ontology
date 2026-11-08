# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn embedded rows into the records :meth:`PostgresVDB.run` writes.

An embedded row is the dict the row builders produce (``text``, ``path``,
``page_number``, ``metadata``) after the embed step has set
``metadata["embedding"]``. The record keeps every top-level metadata key --
retrieval reads hits' ``metadata["id"]``, ``label``, ``name`` and so on from
there -- and adds ``content``, ``content_metadata`` and ``source_metadata``.

The shape matches what ``nemo_retriever``'s ``to_client_vdb_records`` wrote
for these rows, so rows written before and after it was replaced are
indistinguishable in the store.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable
from uuid import NAMESPACE_URL, uuid5

if TYPE_CHECKING:
    import pandas as pd

    from auto_ontology.vdb.postgres import PostgresVDB

__all__ = [
    "projection_id",
    "global_pql_projection_id",
    "to_vdb_records",
    "write_embedded_rows",
]


def projection_id(collection_name: str, connection_id: str, node_id: str) -> str:
    """Stable vector identity; JSON preserves tuple boundaries without ambiguity."""
    if any(
        not isinstance(v, str) or not v.strip()
        for v in (collection_name, connection_id, node_id)
    ):
        raise ValueError(
            "source_required: collection, connection_id and node id are required"
        )
    return str(
        uuid5(
            NAMESPACE_URL,
            json.dumps(
                [collection_name, connection_id, node_id],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )


def global_pql_projection_id(collection_name: str, node_id: str) -> str:
    """Canonical global PQL identity shared by runtime and explicit migration."""
    return projection_id(collection_name, "global:PqlAnalysis", node_id)


def validate_global_pql_metadata(metadata: dict[str, Any]) -> None:
    """Only explicitly global, correctly identified PQL rows bypass source identity."""
    if (
        metadata.get("label") != "PqlAnalysis"
        or metadata.get("connection_id") is not None
        or (metadata.get("content_metadata") or {}).get("connection_id") is not None
        or not isinstance(metadata.get("id"), str)
        or not metadata["id"].strip()
    ):
        raise ValueError(
            "invalid_global_pql: PqlAnalysis requires node id and no connection_id"
        )


def _text(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    """The first non-blank text field, unmodified."""
    for value in (row.get("text"), row.get("content"), metadata.get("content")):
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _page_number(row: dict[str, Any], content_metadata: dict[str, Any]) -> int | None:
    pages = []
    for value in (content_metadata.get("page_number"), row.get("page_number")):
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(pages) if pages else None


def _record(row: dict[str, Any], *, global_pql: bool = False) -> dict[str, Any] | None:
    metadata = dict(row.get("metadata") or {})
    # An empty list is what a failed request leaves behind, so it is as absent
    # as a missing key: writing it would store a row no search can reach.
    if not metadata.get("embedding"):
        return None
    text = _text(row, metadata)
    if not text:
        return None

    if global_pql:
        validate_global_pql_metadata(metadata)
    for key in ("id",) if global_pql else ("id", "connection_id"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"source_required: embedded row requires {key}")

    content_metadata = dict(metadata.get("content_metadata") or {})
    content_metadata["connection_id"] = metadata.get("connection_id")
    page = _page_number(row, content_metadata)
    if page is not None:
        content_metadata["page_number"] = page

    source_metadata = dict(metadata.get("source_metadata") or {})
    source_path = next(
        (
            value.strip()
            for value in (metadata.get("source_path"), row.get("path"))
            if isinstance(value, str) and value.strip()
        ),
        "",
    )
    if source_path:
        source_metadata.setdefault("source_id", source_path)
        source_metadata.setdefault("source_name", PurePosixPath(source_path).name)

    metadata["content"] = text
    metadata["content_metadata"] = content_metadata
    metadata["source_metadata"] = source_metadata
    return {
        "document_type": str(row.get("document_type") or "text"),
        "metadata": metadata,
    }


def to_vdb_records(
    rows: "Iterable[dict[str, Any]] | pd.DataFrame",
    *,
    global_pql: bool = False,
) -> list[list[dict[str, Any]]]:
    """Records for the *rows* that carry an embedding and some text.

    Returns ``[records]``, or ``[]`` when none qualify -- never ``[[]]``, which
    is truthy and would read as "something to write".
    """
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict(orient="records")
    records = [
        record
        for row in rows or []
        if isinstance(row, dict)
        and (record := _record(row, global_pql=global_pql)) is not None
    ]
    return [records] if records else []


def write_embedded_rows(
    vdb: "PostgresVDB", rows: "Iterable[dict[str, Any]] | pd.DataFrame"
) -> int:
    """Write the embedded *rows* to *vdb*; returns the number of rows written."""
    records = to_vdb_records(rows, global_pql=getattr(vdb, "global_pql", False))
    return vdb.run(records) if records else 0
