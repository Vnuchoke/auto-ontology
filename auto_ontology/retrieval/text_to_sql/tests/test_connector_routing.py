# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast

import pytest
from auto_ontology.connectors.base import SQLDatabase

from auto_ontology.retrieval.text_to_sql.connector_routing import (
    resolve_connector_from_tables,
    resolve_target_database_name,
)


def _connector(database_name: str) -> SQLDatabase:
    return cast(SQLDatabase, SimpleNamespace(database_name=database_name))


def test_routes_all_tables_to_their_database() -> None:
    first = _connector("first")
    second = _connector("second")

    resolved = resolve_connector_from_tables(
        [
            {"name": "orders", "database_name": "second"},
            {"name": "customers", "database_name": "second"},
        ],
        [first, second],
    )

    assert resolved is second


def test_rejects_missing_database_with_multiple_connectors() -> None:
    connectors = [_connector("first"), _connector("second")]

    with pytest.raises(ValueError, match="do not identify a database"):
        resolve_connector_from_tables([{"name": "orders"}], connectors)


def test_allows_missing_database_with_one_connector() -> None:
    connector = _connector("only")

    assert resolve_connector_from_tables([{"name": "orders"}], [connector]) is connector


def test_resolves_target_database_to_canonical_connector_name() -> None:
    connectors = [_connector("wideworldimporters")]

    assert (
        resolve_target_database_name("WIDEWORLDIMPORTERS", connectors)
        == "wideworldimporters"
    )


def test_rejects_unknown_target_database() -> None:
    connectors = [_connector("wideworldimporters")]

    with pytest.raises(ValueError, match="does not match a configured database"):
        resolve_target_database_name(
            "42d62c22-b1c4-4ed7-8173-6aec9e2bd432",
            connectors,
        )


@pytest.mark.parametrize("selected_index", [0, 1], ids=["local", "remote"])
def test_routes_same_name_tables_by_connection_id(
    same_name_sources: dict,
    selected_index: int,
) -> None:
    from copy import deepcopy

    sources = same_name_sources["sources"]
    connectors = [
        cast(
            SQLDatabase,
            SimpleNamespace(
                database_name=source["connection"]["database"],
                connection_id=source["connection_id"],
            ),
        )
        for source in sources
    ]
    selected = connectors[selected_index]
    neighbor = connectors[1 - selected_index]
    tables_before = deepcopy([source["table"] for source in sources])
    # Put the neighbor last: the current name-keyed dictionary picks it incorrectly.
    ordered = [selected, neighbor]
    assert (
        resolve_connector_from_tables(
            [sources[1 - selected_index]["table"]],
            ordered,
        )
        is neighbor
    )
    assert (
        resolve_connector_from_tables(
            [sources[selected_index]["table"]],
            ordered,
        )
        is selected
    )
    assert [source["table"] for source in sources] == tables_before


def test_rejects_mixed_connection_id_dependencies_with_same_database_name(
    same_name_sources: dict,
) -> None:
    sources = same_name_sources["sources"]
    connectors = [
        cast(
            SQLDatabase,
            SimpleNamespace(
                database_name=source["connection"]["database"],
                connection_id=source["connection_id"],
            ),
        )
        for source in sources
    ]
    assert (
        resolve_connector_from_tables([sources[1]["table"]], connectors)
        is connectors[1]
    )
    with pytest.raises(
        ValueError, match=r"(?i)(multiple|mixed|cross).*(connection|source)"
    ):
        resolve_connector_from_tables(
            [source["table"] for source in sources], connectors
        )
