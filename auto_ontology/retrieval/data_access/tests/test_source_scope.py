# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.data_access.semantic_search import search_semantic_index


def test_source_filter_precedes_query_and_drops_foreign_hits() -> None:
    retriever = SimpleNamespace(
        vdb_kwargs={"vdb": SimpleNamespace(metadata_filter_format="dict")},
        query=Mock(
            return_value=[
                {
                    "text": "A",
                    "metadata": {
                        "id": "node-a",
                        "label": "Column",
                        "connection_id": "a",
                    },
                },
                {
                    "text": "B",
                    "metadata": {
                        "id": "node-b",
                        "label": "Column",
                        "connection_id": "b",
                    },
                },
                {"text": "legacy", "metadata": {"id": "node-null", "label": "Column"}},
            ]
        ),
    )
    rows = search_semantic_index(retriever, "status", ["Column"], connection_id="a")
    assert [r["id"] for r in rows] == ["node-a"]
    assert rows[0]["connection_id"] == "a"
    assert retriever.query.call_args.kwargs["vdb_kwargs"]["where"] == {
        "connection_id": "a",
        "label": "Column",
    }


def test_missing_or_conflicting_scope_never_queries() -> None:
    retriever = SimpleNamespace(
        vdb_kwargs={"vdb": SimpleNamespace(connection_id="a")}, query=Mock()
    )
    with pytest.raises(ValueError, match="source_required"):
        search_semantic_index(retriever, "status", ["Column"])
    with pytest.raises(ValueError, match="source_conflict"):
        search_semantic_index(retriever, "status", ["Column"], connection_id="b")
    retriever.query.assert_not_called()


def test_only_verified_global_pql_client_can_omit_source() -> None:
    from auto_ontology.vdb.postgres import PostgresVDB

    vdb = object.__new__(PostgresVDB)
    vdb.global_pql = True
    vdb.connection_id = None
    retriever = SimpleNamespace(vdb_kwargs={"vdb": vdb}, query=Mock(return_value=[]))
    assert search_semantic_index(retriever, "example", ["PqlAnalysis"]) == []
    assert retriever.query.call_args.kwargs["vdb_kwargs"]["where"] == {
        "label": "PqlAnalysis"
    }
    with pytest.raises(ValueError):
        search_semantic_index(retriever, "example", ["Column"])


def test_real_client_schema_gate_precedes_embedding_query() -> None:
    from auto_ontology.vdb.postgres import PostgresVDB

    vdb = object.__new__(PostgresVDB)
    vdb.global_pql = False
    vdb.connection_id = None
    vdb._require_source_schema = Mock(side_effect=ValueError("source_not_ready"))
    retriever = SimpleNamespace(vdb_kwargs={"vdb": vdb}, query=Mock())
    with pytest.raises(ValueError, match="source_not_ready"):
        search_semantic_index(retriever, "status", ["Column"], connection_id="a")
    retriever.query.assert_not_called()
