# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.dal import model_interchange as dal
from auto_ontology.server.model_interchange import embed
from auto_ontology.server.model_interchange.schemas import AutoOntologyModelDocument


def test_shared_existing_term_gets_each_represented_column_and_sql_projection(
    monkeypatch,
):
    document = AutoOntologyModelDocument.model_validate(
        {
            "data_layer": {
                "databases": [
                    {
                        "id": cid,
                        "display_label": f"Label {cid}",
                        "schemas": [
                            {
                                "id": f"s-{cid}",
                                "name": "public",
                                "database_name": "shop",
                                "tables": [
                                    {
                                        "id": f"t-{cid}",
                                        "name": "orders",
                                        "columns": [
                                            {"id": f"c-{cid}", "name": "amount"}
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                    for cid in ("A", "B", "C")
                ]
            },
            "semantic_layer": {
                "terms": [
                    {
                        "id": "term",
                        "name": "Revenue",
                        "represents": ["t-A"],
                        "columns_attributes": [
                            {"id": "attr-B", "name": "Amount", "column_id": "c-B"}
                        ],
                    }
                ],
                "sql_attributes": {
                    "manual": [
                        {
                            "id": "sql-C",
                            "name": "Revenue",
                            "connection_id": "C",
                            "term_id": "term",
                            "sql": "SELECT 1",
                        }
                    ]
                },
            },
        }
    )
    monkeypatch.setattr(
        dal, "_resolve_entities_batch", lambda *_args: {"term": ("live-term", False)}
    )
    monkeypatch.setattr(dal, "store", Mock())
    monkeypatch.setattr(dal, "_link", Mock())
    monkeypatch.setattr(
        dal,
        "_database_names_for_ids",
        lambda _ids: {cid: "shop" for cid in ("A", "B", "C")},
    )
    buffer = embed.ImportEmbedBuffer()
    dal._import_terms(
        document,
        {key: key for key in ("A", "B", "C", "t-A")},
        {"terms": 0},
        {"terms": 0},
        buffer,
    )
    assert {row["metadata"]["connection_id"] for row in buffer.semantic_rows} == {
        "A",
        "B",
        "C",
    }
    assert {row["metadata"]["id"] for row in buffer.semantic_rows} == {"live-term"}
    assert {row["text"] for row in buffer.semantic_rows} == {"Term: Revenue."}
    assert all(
        row["metadata"]["database_name"] == "shop" for row in buffer.semantic_rows
    )


def test_import_replaces_only_selected_projection_after_successful_embedding(
    monkeypatch,
):
    events = []
    row = embed.build_custom_analysis_semantic_row(
        connection_id="A",
        live_id="node",
        name="n",
        description="local draft",
        sql="SELECT 1",
        database_name="shop",
    )
    monkeypatch.setattr(embed, "get_embed_params", lambda: None)
    monkeypatch.setattr(
        embed,
        "embed_rows",
        lambda rows, *_args, **_kwargs: events.append("embed") or rows,
    )
    monkeypatch.setattr(
        embed, "write_embedded_rows", lambda _vdb, rows: events.append("write")
    )
    vdb = Mock()
    vdb.delete_by_id.side_effect = lambda node, **kwargs: events.append((node, kwargs))
    for _ in range(2):
        assert embed._embed_and_ingest_rows([row], vdb) == 1
    assert events == ["embed", ("node", {"connection_id": "A"}), "write"] * 2


def test_import_embedding_failure_keeps_previous_projection(monkeypatch):
    row = embed.build_sql_attribute_semantic_row(
        connection_id="A",
        live_id="node",
        name="n",
        description="",
        term_name="Revenue",
        sql="SELECT 1",
        database_name="shop",
    )
    monkeypatch.setattr(embed, "get_embed_params", lambda: None)
    monkeypatch.setattr(embed, "embed_rows", Mock(side_effect=RuntimeError("offline")))
    vdb = Mock()
    with pytest.raises(RuntimeError, match="offline"):
        embed._embed_and_ingest_rows([row], vdb)
    vdb.delete_by_id.assert_not_called()


def test_import_column_projection_contains_only_its_samples():
    rows = embed.build_column_attribute_semantic_rows(
        connection_id="A",
        database_name="shop",
        live_id="attr",
        name="Amount",
        description="local-A",
        term_name="Revenue",
        source_column="amount",
        table_id="table-A",
        table_name="orders",
        is_unique=False,
        sample_values=["sample-A"],
        schema_name="public",
    )
    assert len(rows) == 1
    assert rows[0]["metadata"]["connection_id"] == "A"
    assert rows[0]["metadata"]["content_metadata"]["connection_id"] == "A"
    assert "sample-A" in rows[0]["text"]
