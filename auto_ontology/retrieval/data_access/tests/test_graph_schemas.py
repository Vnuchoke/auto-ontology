# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.retrieval.data_access import graph_schemas


def _row(source: str) -> dict:
    return {
        "connection_id": source,
        "schema_id": f"schema-{source}",
        "database_name": "sales",
        "table_schema": "public",
        "table_name": "orders",
        "table_id": f"table-{source}",
        "column_name": "amount",
        "column_id": f"column-{source}",
        "data_type": "int",
    }


def test_snapshot_preserves_registered_root_and_child_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_schemas, "fetch_schemas_by_ids", lambda ids, **kw: [_row("a")]
    )
    schema = graph_schemas.get_schemas_by_ids(["schema-a"], connection_id="a")["public"]
    assert schema.get_db_node().get_id() == "a"
    assert schema.get_db_node().get_name() == "sales"
    assert schema.get_schema_node().get_id() == "schema-a"
    assert schema.tables_df.id.tolist() == ["table-a"]
    assert schema.columns_df.id.tolist() == ["column-a"]


def test_snapshot_cannot_merge_same_named_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_schemas, "fetch_schemas_by_ids", lambda ids, **kw: [_row("a"), _row("b")]
    )
    with pytest.raises(ValueError, match="source"):
        graph_schemas.get_schemas_by_ids(["schema-a", "schema-b"], connection_id="a")


def test_empty_schema_selection_does_not_expand_to_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = Mock(side_effect=AssertionError("empty selection must not read all sources"))
    monkeypatch.setattr(graph_schemas, "fetch_schemas_by_ids", read)
    assert graph_schemas.get_schemas_by_ids([], connection_id="a") == {}
    read.assert_not_called()


def test_missing_source_does_not_read_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    read = Mock(side_effect=AssertionError("no global query context"))
    monkeypatch.setattr(graph_schemas, "fetch_schemas_by_ids", read)
    with pytest.raises(ValueError, match="source_required"):
        graph_schemas.get_schemas_by_ids(["schema-a"])
    read.assert_not_called()
