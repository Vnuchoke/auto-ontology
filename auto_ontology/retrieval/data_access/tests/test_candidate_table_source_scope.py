# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.dal import datasources
from auto_ontology.retrieval.data_access.relevant_tables import (
    get_relevant_tables_from_candidates,
)


def test_scoped_candidate_tables_hydrate_by_node_id_without_trusting_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = Mock(
        return_value=[
            {
                "id": "table-a",
                "connection_id": "a",
                "name": "orders",
                "database_name": "sales",
                "schema_name": "public",
                "description": "verified A",
                "columns": [],
            }
        ]
    )
    monkeypatch.setattr(datasources, "fetch_tables_by_ids", fetch)
    candidates = [
        {
            "relevant_tables": [
                {"id": "table-a", "connection_id": "a", "description": "forged text"},
                {"id": "table-b", "connection_id": "a", "description": "foreign text"},
            ]
        }
    ]
    result = get_relevant_tables_from_candidates(candidates, connection_id="a")
    fetch.assert_called_once_with(["table-a", "table-b"], connection_id="a")
    assert result[0]["connection_id"] == "a"
    assert result[0]["description"] == "verified A"
    assert "foreign text" not in str(result) and "forged text" not in str(result)
    assert candidates == [{}]


def test_blank_source_does_not_hydrate_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = Mock(side_effect=AssertionError("must not hydrate"))
    monkeypatch.setattr(datasources, "fetch_tables_by_ids", fetch)
    assert (
        get_relevant_tables_from_candidates(
            [{"relevant_tables": [{"id": "table-b", "connection_id": "a"}]}],
            connection_id=" ",
        )
        == []
    )
    fetch.assert_not_called()


def test_admin_candidate_table_extraction_retains_existing_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = Mock(side_effect=AssertionError("admin extraction remains pure"))
    monkeypatch.setattr(datasources, "fetch_tables_by_ids", fetch)
    result = get_relevant_tables_from_candidates(
        [
            {
                "relevant_tables": [
                    {"id": "table", "name": "orders", "description": "admin"}
                ]
            }
        ]
    )
    assert result[0]["id"] == "table" and result[0]["description"] == "admin"
    fetch.assert_not_called()
