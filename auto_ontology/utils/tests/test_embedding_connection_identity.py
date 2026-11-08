# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp, build_embed_row
from auto_ontology.utils import embedding


def test_catalog_filters_connection_before_rendering_same_named_objects():
    tables = pd.DataFrame(
        [
            dict(
                id=f"t-{cid}",
                connection_id=cid,
                table_schema="public",
                table_name="orders",
                description=f"draft-{cid}",
            )
            for cid in ("A", "B")
        ]
    )
    columns = pd.DataFrame(
        [
            dict(
                id=f"c-{cid}",
                connection_id=cid,
                table_schema="public",
                table_name="orders",
                column_name=f"col-{cid}",
                description=f"private-{cid}",
                data_type="text",
                sample_values=[f"sample-{cid}"],
            )
            for cid in ("A", "B")
        ]
    )
    rows = (
        CatalogEmbeddingRowsOp(database_name="shop", connection_id="A")
        .process((tables, columns))
        .to_dict("records")
    )
    assert {row["metadata"]["id"] for row in rows} == {"t-A", "c-A"}
    for row in rows:
        assert row["metadata"]["connection_id"] == "A"
        assert row["metadata"]["content_metadata"]["connection_id"] == "A"
        assert "-B" not in row["text"]
        assert "database_name: shop" in row["text"]


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame([dict(id="t")]), pd.DataFrame([dict(id="t", connection_id=None)])],
)
def test_catalog_rejects_unowned_nonempty_frames(frame):
    with pytest.raises(ValueError, match="connection_id"):
        CatalogEmbeddingRowsOp(database_name="shop", connection_id="A").process(
            (frame, pd.DataFrame())
        )


def test_row_preserves_node_identity_and_physical_name():
    row = build_embed_row(
        text="unchanged",
        node_id="node",
        label="Table",
        name="orders",
        schema_name="public",
        database_name="shop",
        connection_id="A",
    )
    assert row["text"] == "unchanged"
    assert row["metadata"]["id"] == "node"
    assert row["metadata"]["database_name"] == "shop"
    assert row["metadata"]["connection_id"] == "A"


def test_docs_pass_validated_identity_to_embedding_and_storage(monkeypatch):
    captured = []
    monkeypatch.setattr(
        embedding,
        "embed_rows",
        lambda rows, *_args, **_kwargs: captured.extend(rows) or rows,
    )
    monkeypatch.setattr(embedding, "write_embedded_rows", lambda _vdb, rows: len(rows))
    assert (
        embedding.embed_docs_into_vdb(
            [
                dict(
                    id="node",
                    name="n",
                    label="SqlAttribute",
                    text="local draft",
                    connection_id="A",
                )
            ],
            None,
            object(),
            "shop",
            connection_id="A",
        )
        == 1
    )
    metadata = captured[0]["metadata"]
    assert metadata["id"] == "node"
    assert (
        metadata["connection_id"]
        == metadata["content_metadata"]["connection_id"]
        == "A"
    )
    assert metadata["database_name"] == "shop"


@pytest.mark.parametrize("cid", [None, "B"])
def test_docs_do_not_assign_selected_identity_to_unknown_or_foreign_docs(
    monkeypatch, cid
):
    monkeypatch.setattr(
        embedding,
        "embed_rows",
        lambda *_args, **_kwargs: pytest.fail("untrusted text reached embedding"),
    )
    with pytest.raises(ValueError, match="connection_id"):
        embedding.embed_docs_into_vdb(
            [dict(id="node", text="private", connection_id=cid)],
            None,
            object(),
            connection_id="A",
        )


def test_rows_reject_legacy_before_embedding(monkeypatch):
    monkeypatch.setattr(
        embedding,
        "embed_rows",
        lambda *_args, **_kwargs: pytest.fail("legacy text reached embedding"),
    )
    with pytest.raises(ValueError, match="connection_id"):
        embedding.embed_rows_into_vdb(
            [dict(text="legacy", metadata={"id": "node"})], None, object(), what="test"
        )
