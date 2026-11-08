# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest

from auto_ontology.dal import custom_analyses as ca
from auto_ontology.server.custom_analyses import service as analyses
from auto_ontology.server.datasources import service as datasources
from auto_ontology.server.sql_attributes import service as attributes
from auto_ontology.utils import embedding
from auto_ontology.vdb import records
import auto_ontology.utils as utils
import auto_ontology.vdb as vdb_module


@pytest.fixture
def captured_embedding(monkeypatch):
    captured = []
    monkeypatch.setattr(utils, "get_embed_params", lambda: None)
    monkeypatch.setattr(attributes, "get_embed_params", lambda: None)
    monkeypatch.setattr(
        embedding,
        "embed_rows",
        lambda rows, *_args, **_kwargs: captured.extend(rows) or rows,
    )
    monkeypatch.setattr(records, "write_embedded_rows", lambda _vdb, rows: len(rows))
    return captured


def test_catalog_refresh_repeatedly_deletes_only_selected_source(
    monkeypatch, captured_embedding
):
    tables = pd.DataFrame(
        [
            dict(
                id="table",
                connection_id="A",
                table_name="orders",
                table_schema="public",
                description="local-A",
            )
        ]
    )
    columns = pd.DataFrame(
        [
            dict(
                id="column",
                connection_id="A",
                table_name="orders",
                table_schema="public",
                column_name="amount",
                description="draft-A",
                data_type="int",
                sample_values=["sample-A"],
            )
        ]
    )
    monkeypatch.setattr(
        datasources,
        "fetch_tables_and_columns_by_node_ids",
        lambda _ids: (tables, columns, "shop"),
    )
    vdb = Mock()
    factory = Mock(return_value=vdb)
    monkeypatch.setattr(vdb_module, "get_data_vdb", factory)
    for _ in range(2):
        datasources._refresh_vdb_embeddings(["table", "column"])
    assert len(captured_embedding) == 4
    assert {row["metadata"]["connection_id"] for row in captured_embedding} == {"A"}
    assert all(
        call.kwargs == {"connection_id": "A"}
        for call in vdb.delete_by_id.call_args_list
    )
    assert all(call.kwargs == {"connection_id": "A"} for call in factory.call_args_list)
    assert "sample-A" in captured_embedding[1]["text"]


def test_catalog_refresh_rejects_mixed_source_input_before_embedding(
    monkeypatch, captured_embedding
):
    tables = pd.DataFrame([dict(id="table", connection_id="A")])
    columns = pd.DataFrame([dict(id="column", connection_id="B")])
    monkeypatch.setattr(
        datasources,
        "fetch_tables_and_columns_by_node_ids",
        lambda _ids: (tables, columns, "shop"),
    )
    with pytest.raises(ValueError, match="single connection_id"):
        datasources._refresh_vdb_embeddings(["table", "column"])
    assert captured_embedding == []


def test_sql_attribute_refresh_passes_identity_to_dal_and_scoped_delete(
    monkeypatch, captured_embedding
):
    fetch = Mock(
        return_value=[
            dict(
                id="attr",
                name="Amount",
                label="SqlAttribute",
                text="draft-A",
                source="manual",
                connection_id="A",
            )
        ]
    )
    monkeypatch.setattr(attributes, "fetch_sql_attribute_docs", fetch)
    vdb = Mock()
    monkeypatch.setattr(attributes, "get_semantic_vdb", Mock(return_value=vdb))
    attributes._reembed_sql_attribute("attr", "shop", connection_id="A")
    fetch.assert_called_once_with("attr", connection_id="A")
    vdb.delete_by_id.assert_called_once_with("attr", connection_id="A")
    assert captured_embedding[0]["metadata"]["connection_id"] == "A"
    assert captured_embedding[0]["metadata"]["source"] == "manual"


def test_custom_analysis_refresh_uses_owned_identity(monkeypatch, captured_embedding):
    fetch = Mock(
        return_value=[
            dict(
                id="analysis",
                name="Saved",
                label="CustomAnalysis",
                text="local-A",
                connection_id="A",
            )
        ]
    )
    monkeypatch.setattr(ca, "_custom_analysis_docs", fetch)
    vdb = Mock()
    ca.embed_custom_analyses(None, vdb, "analysis", "shop", connection_id="A")
    fetch.assert_called_once_with("analysis", connection_id="A")
    vdb.delete_by_id.assert_called_once_with("analysis", connection_id="A")
    assert captured_embedding[0]["metadata"]["content_metadata"]["connection_id"] == "A"


def test_analysis_delete_reads_identity_before_deleting_definition(monkeypatch):
    events = []
    monkeypatch.setattr(analyses, "get_custom_analysis_by_id", lambda _id: "analysis")
    monkeypatch.setattr(
        ca,
        "get_custom_analysis_connection_id",
        lambda _id: events.append("read") or "A",
    )
    monkeypatch.setattr(
        analyses, "delete_custom_analysis_node", lambda _id: events.append("delete")
    )
    vdb = Mock()
    monkeypatch.setattr(vdb_module, "get_semantic_vdb", Mock(return_value=vdb))
    assert analyses.delete_custom_analysis("analysis") == {"id": "analysis"}
    assert events == ["read", "delete"]
    vdb.delete_by_id.assert_called_once_with("analysis", connection_id="A")


def test_legacy_analysis_is_not_embedded(monkeypatch):
    monkeypatch.setattr(ca, "get_custom_analysis_connection_id", lambda _id: None)
    embed = Mock()
    monkeypatch.setattr(analyses, "embed_custom_analyses", embed)
    analyses._embed_analysis("legacy")
    embed.assert_not_called()


def test_direct_column_attribute_refresh_uses_each_dal_projection_identity(monkeypatch):
    contexts = [
        (
            dict(connection_id=cid, database_name="shop"),
            [dict(id="attribute", connection_id=cid, table_id=f"table-{cid}")],
        )
        for cid in ("A", "B")
    ]
    monkeypatch.setattr(
        datasources,
        "fetch_column_attribute_embedding_contexts_by_column_id",
        lambda _id: contexts,
    )
    embedders = [Mock(), Mock()]
    factory = Mock(side_effect=embedders)
    monkeypatch.setattr(datasources, "build_semantic_embedder", factory)
    datasources._refresh_semantic_column_attribute_embeddings("column")
    for index, cid in enumerate(("A", "B")):
        assert factory.call_args_list[index].kwargs == {
            "connection_id": cid,
            "reset": False,
        }
        embedders[index].vdb.delete_by_id.assert_called_once_with(
            "attribute", connection_id=cid
        )
        embedders[index].embed_column_attributes.assert_called_once_with(
            contexts[index][1]
        )
