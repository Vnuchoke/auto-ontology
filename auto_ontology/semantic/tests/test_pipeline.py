# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flat table-by-table compile_semantic_layer loop."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from auto_ontology.connectors.sources import SourceDescriptor

from auto_ontology.semantic.embed import SemanticEmbedder
from auto_ontology.semantic.models import ProcessTableResult
from auto_ontology.semantic.pipeline import compile_semantic_layer
from auto_ontology.vdb.postgres import PostgresVDB


@pytest.fixture(autouse=True)
def active_source(monkeypatch):
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        lambda: [SourceDescriptor("dbx", "Test", "sales", "postgres", "managed")],
    )


@patch("auto_ontology.semantic.pipeline.process_table")
@patch("auto_ontology.semantic.pipeline.fetch_table_context")
@patch("auto_ontology.semantic.pipeline.load_domain_summary", return_value=None)
@patch("auto_ontology.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_processes_all_tables(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    mock_process: MagicMock,
) -> None:
    """Two tables → two process_table calls → count == 2."""
    mock_fetch_tables.return_value = [
        {"id": "t1", "name": "orders", "description": "", "connection_id": "dbx"},
        {"id": "t2", "name": "customers", "description": "", "connection_id": "dbx"},
    ]
    mock_fetch_ctx.side_effect = [
        {"columns": [{"name": "amount"}], "fks": [], "connection_id": "dbx"},
        {"columns": [{"name": "name"}], "fks": [], "connection_id": "dbx"},
    ]

    count = compile_semantic_layer("dbx")

    assert count == 2
    assert mock_process.call_count == 2


@pytest.mark.parametrize("collection_exists", [False, True])
@pytest.mark.parametrize("has_tables", [False, True])
def test_compile_initializes_semantic_collection_only_from_embedded_rows(
    monkeypatch: pytest.MonkeyPatch, collection_exists: bool, has_tables: bool
) -> None:
    vdb = PostgresVDB(
        connection_string="postgresql://unused/test",
        collection_name="semantic_objects_layer",
        connection_id="dbx",
    )
    engine = MagicMock()
    store = MagicMock()
    monkeypatch.setattr(vdb, "_table_exists", lambda: collection_exists)
    monkeypatch.setattr(vdb, "_get_engine", lambda: engine)
    monkeypatch.setattr(vdb, "_ensure_schema", MagicMock())
    monkeypatch.setattr(vdb, "_require_source_schema", MagicMock())
    monkeypatch.setattr(vdb, "_create_vector_index", MagicMock())
    monkeypatch.setattr("auto_ontology.vdb.postgres.psycopg.connect", MagicMock())
    monkeypatch.setattr(
        "auto_ontology.vdb.postgres.PGVectorStore.create_sync",
        MagicMock(return_value=store),
    )
    embedder = SemanticEmbedder("sales", "dbx", MagicMock(), vdb)
    tables = [{"id": "t1", "name": "orders"}] if has_tables else []
    monkeypatch.setattr(
        "auto_ontology.semantic.pipeline.fetch_all_tables_without_term",
        lambda **kwargs: tables,
    )
    monkeypatch.setattr(
        "auto_ontology.semantic.pipeline.fetch_table_context",
        lambda *args, **kwargs: {"columns": [{"name": "id"}]},
    )
    monkeypatch.setattr(
        "auto_ontology.semantic.pipeline.is_distinct_value_probing_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "auto_ontology.semantic.pipeline.load_domain_summary", lambda _: None
    )

    def batch_embed(rows: list[dict[str, Any]], _params: Any) -> pd.DataFrame:
        assert vdb.vector_size is None
        engine.init_vectorstore_table.assert_not_called()
        for row in rows:
            row["metadata"]["embedding"] = [0.25] * 1024
        return pd.DataFrame(rows)

    def process_table(
        _table: dict[str, Any], _ctx: dict[str, Any], **kwargs: Any
    ) -> ProcessTableResult:
        with kwargs["commit_slot"]:
            assert (
                kwargs["embedder"].embed_term(
                    {"id": "term1", "name": "Order", "connection_id": "dbx"}, []
                )
                == 1
            )
        return ProcessTableResult(term_names=["Order"])

    monkeypatch.setattr("auto_ontology.semantic.embed.batch_embed", batch_embed)
    monkeypatch.setattr("auto_ontology.semantic.pipeline.process_table", process_table)

    assert compile_semantic_layer("dbx", embedder=embedder) == int(has_tables)
    assert vdb.vector_size == (1024 if has_tables else None)
    assert store.add_embeddings.call_count == int(has_tables)
    if has_tables:
        assert store.add_embeddings.call_args.kwargs["embeddings"] == [[0.25] * 1024]
    if has_tables and not collection_exists:
        engine.init_vectorstore_table.assert_called_once()
        assert engine.init_vectorstore_table.call_args.kwargs["vector_size"] == 1024
    else:
        engine.init_vectorstore_table.assert_not_called()


@patch("auto_ontology.semantic.pipeline.fetch_table_context")
@patch("auto_ontology.semantic.pipeline.load_domain_summary", return_value=None)
@patch("auto_ontology.semantic.pipeline.fetch_all_tables_without_term")
def test_term_commits_follow_canonical_order_when_llm_completion_is_reversed(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
) -> None:
    mock_fetch_tables.return_value = [
        {"id": "z-id", "name": "zeta", "schema_name": "sales", "connection_id": "dbx"},
        {"id": "a-id", "name": "alpha", "schema_name": "sales", "connection_id": "dbx"},
    ]
    mock_fetch_ctx.return_value = {
        "columns": [{"name": "id"}],
        "fks": [],
        "connection_id": "dbx",
    }
    zeta_ready = threading.Event()
    commits: list[str] = []

    def fake_process(
        table: dict[str, Any], _ctx: dict[str, Any], **kwargs: Any
    ) -> ProcessTableResult:
        if table["name"] == "alpha":
            assert zeta_ready.wait(timeout=2)
        else:
            zeta_ready.set()
        with kwargs["commit_slot"]:
            commits.append(table["name"])
        return ProcessTableResult()

    with patch(
        "auto_ontology.semantic.pipeline.process_table", side_effect=fake_process
    ):
        count = compile_semantic_layer("dbx")

    assert count == 2
    assert commits == ["alpha", "zeta"]


@patch("auto_ontology.semantic.pipeline.process_table")
@patch("auto_ontology.semantic.pipeline.fetch_table_context")
@patch("auto_ontology.semantic.pipeline.load_domain_summary", return_value=None)
@patch("auto_ontology.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_skips_columnless_table(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    mock_process: MagicMock,
) -> None:
    """Table with no columns is skipped without calling process_table."""
    mock_fetch_tables.return_value = [
        {"id": "t1", "name": "empty_table", "description": "", "connection_id": "dbx"}
    ]
    mock_fetch_ctx.return_value = {"columns": [], "fks": [], "connection_id": "dbx"}

    count = compile_semantic_layer("dbx")

    assert count == 0
    mock_process.assert_not_called()


@patch(
    "auto_ontology.semantic.pipeline.process_table", side_effect=RuntimeError("boom")
)
@patch("auto_ontology.semantic.pipeline.fetch_table_context")
@patch("auto_ontology.semantic.pipeline.load_domain_summary", return_value=None)
@patch("auto_ontology.semantic.pipeline.fetch_all_tables_without_term")
def test_compile_continues_after_error(
    mock_fetch_tables: MagicMock,
    _mock_summary: MagicMock,
    mock_fetch_ctx: MagicMock,
    _mock_process: MagicMock,
) -> None:
    """process_table raising an exception should not abort the loop."""
    mock_fetch_tables.return_value = [
        {"id": "t1", "name": "orders", "description": "", "connection_id": "dbx"}
    ]
    mock_fetch_ctx.return_value = {
        "columns": [{"name": "x"}],
        "fks": [],
        "connection_id": "dbx",
    }

    count = compile_semantic_layer("dbx")

    assert count == 0
