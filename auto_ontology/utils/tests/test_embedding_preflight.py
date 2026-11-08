# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.utils import embedding
from auto_ontology.vdb.postgres import PostgresVDB


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    embed = Mock(side_effect=AssertionError("unvalidated batch reached embedding"))
    write = Mock(side_effect=AssertionError("unvalidated batch reached storage"))
    monkeypatch.setattr(embedding, "embed_rows", embed)
    monkeypatch.setattr(embedding, "write_embedded_rows", write)
    return embed, write


@pytest.mark.parametrize(
    "bound,metadata,error",
    [
        ("A", {"id": "node", "connection_id": "B"}, "source_conflict"),
        ("B", {"id": "node", "connection_id": "A"}, "source_conflict"),
        (
            "A",
            {"id": "node", "connection_id": "A", "label": "PqlAnalysis"},
            "invalid_global_pql",
        ),
        *[
            (None, {"id": "node", "connection_id": value}, "connection_id")
            for value in (None, "", " ", 0, 1, True, [], {})
        ],
        (None, {"id": "node"}, "connection_id"),
        *[
            (None, {"id": value, "connection_id": "A"}, "id")
            for value in (None, "", " ", 0, 1, True, [], {})
        ],
        (None, {"connection_id": "A"}, "id"),
    ],
)
def test_source_batch_rejected_before_embedding(
    spies: tuple[Mock, Mock], bound: str | None, metadata: dict, error: str
) -> None:
    vdb = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", connection_id=bound
    )
    rows = [
        {
            "text": "valid first",
            "metadata": {"id": "first", "connection_id": bound or "A"},
        },
        {"text": "private", "metadata": metadata},
    ]
    with pytest.raises(ValueError, match=error):
        embedding.embed_rows_into_vdb(rows, None, vdb, what="test")
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"label": "Term"},
        {"connection_id": "A"},
        {"content_metadata": {"connection_id": "A"}},
        {"id": None},
        {"id": " "},
        {"id": 1},
    ],
)
def test_global_batch_rejected_before_embedding(
    spies: tuple[Mock, Mock], metadata: dict
) -> None:
    vdb = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", global_pql=True
    )
    rows = [
        {"text": "valid first", "metadata": {"id": "first", "label": "PqlAnalysis"}},
        {
            "text": "private",
            "metadata": {"id": "node", "label": "PqlAnalysis", **metadata},
        },
    ]
    with pytest.raises(ValueError, match="invalid_global_pql"):
        embedding.embed_rows_into_vdb(rows, None, vdb, what="test")
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize("mode", ["bound", "unbound", "global"])
def test_valid_rows_reach_embedding_and_storage_unchanged(
    spies: tuple[Mock, Mock], mode: str
) -> None:
    vdb = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused",
        connection_id="A" if mode == "bound" else None,
        global_pql=mode == "global",
    )
    metadata = (
        {"id": "node", "label": "PqlAnalysis"}
        if mode == "global"
        else {"id": "node", "label": "Term", "connection_id": "A"}
    )
    rows = [{"text": "valid text", "metadata": metadata}]
    embedded = [{"text": "valid text", "metadata": {**metadata, "embedding": [1.0]}}]
    embed, write = spies
    embed.side_effect = write.side_effect = None
    embed.return_value = embedded
    write.return_value = 1
    assert embedding.embed_rows_into_vdb(rows, None, vdb, what="test") == 1
    embed.assert_called_once_with(rows, None, what="test")
    write.assert_called_once_with(vdb, embedded)
