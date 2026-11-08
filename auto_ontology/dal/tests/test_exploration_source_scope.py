# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.dal import exploration, schema as s
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_term_projections import (
    identity_database as identity_database,
    projections as projections,
)


@pytest.mark.usefixtures("projections")
def test_source_graph_counts_only_source_attributes() -> None:
    graph = exploration.fetch_semantic_exploration_graph(database_id="a")
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["column_attributes_count"] == 1
    assert (
        exploration.fetch_semantic_exploration_graph()["nodes"][0][
            "column_attributes_count"
        ]
        == 2
    )


@pytest.mark.usefixtures("projections")
def test_source_graph_relationship_degree_does_not_disclose_neighbor_source() -> None:
    store().query_write(
        s.term.insert().values(id="foreign", name="Foreign", source="semantic")
    )
    store().query_write(
        s.table__term.insert().values(table_id="t-b", term_id="foreign")
    )
    graph = exploration.fetch_semantic_exploration_graph(database_id="a")
    assert [node["id"] for node in graph["nodes"]] == ["shared"]
    assert graph["nodes"][0]["relationship_count"] == 0
    assert graph["links"] == []


@pytest.mark.usefixtures("projections")
def test_source_graph_keeps_its_zone_chips_without_neighbor_zones() -> None:
    from auto_ontology.dal.zones import create_zone

    a = create_zone(name="A zone", description="", color="#ffffff", item_ids=["t-a"])
    b = create_zone(name="B zone", description="", color="#000000", item_ids=["t-b"])
    global_node = exploration.fetch_semantic_exploration_graph()["nodes"][0]
    assert {zone["id"] for zone in global_node["zones"]} == {a["id"], b["id"]}
    local_node = exploration.fetch_semantic_exploration_graph(database_id="a")["nodes"][
        0
    ]
    assert {zone["id"] for zone in local_node["zones"]} == {a["id"]}
