# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.dal import attributes, schema as s
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    identity_database as identity_database,
)
from auto_ontology.dal.tests.test_query_join_source_scope import (
    join_sources as join_sources,
)


@pytest.fixture
def junction_sources(join_sources: None) -> None:
    store().query_write(s.column__semantic_fk.delete())
    store().query_write(
        s.catalog_table.insert().values(id="target-b", schema_id="s-b", name="target")
    )
    store().query_write(
        s.catalog_column.insert().values(
            id="c-target-b", table_id="target-b", name="id"
        )
    )
    store().query_write(
        s.column__foreign_key.insert().values(
            source_column_id="c-hub-b", target_column_id="c-target-b"
        )
    )
    store().query_write(
        s.catalog_table.update()
        .where(s.catalog_table.c.id.in_(["left", "right", "hub-b"]))
        .values(is_junction_table=True)
    )
    store().query_write(
        s.column__foreign_key.insert().values(
            source_column_id="c-left", target_column_id="c-hub-a"
        )
    )
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id="c-right", attribute_id="attr-a", use_in_join_paths=True
        )
    )
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id="c-hub-b", attribute_id="attr-b", use_in_join_paths=True
        )
    )
    store().query_write(
        s.column__foreign_key.insert().values(
            source_column_id="c-hub-b", target_column_id="c-hub-b"
        )
    )


@pytest.mark.usefixtures("junction_sources")
def test_selected_source_keeps_declared_and_semantic_paths_and_hydration() -> None:
    junctions, paths = attributes.find_connected_junction_tables(
        ["hub-a", "hub-b", "target-b", "hub-a", ""], connection_id="a"
    )
    assert junctions == [
        {"id": "left", "name": "left"},
        {"id": "right", "name": "right"},
    ]
    assert paths == [
        [
            {
                "source_database": "sales",
                "source_schema": "public",
                "source_table": name,
                "source_column": "id",
                "target_database": "sales",
                "target_schema": "public",
                "target_table": "orders",
                "target_column": "id",
            }
        ]
        for name in ("left", "right")
    ]
    assert attributes.find_connected_junction_tables(["target-b"])[0] == [
        {"id": "hub-b", "name": "orders"}
    ]
    assert attributes.find_connected_junction_tables(
        ["hub-a", "target-b"], connection_id="b"
    )[0] == [{"id": "hub-b", "name": "orders"}]


@pytest.mark.usefixtures("junction_sources")
@pytest.mark.parametrize("connection_id", ["b", "unknown", ""])
def test_wrong_source_or_empty_source_has_no_local_paths(connection_id: str) -> None:
    assert attributes.find_connected_junction_tables(
        ["hub-a"], connection_id=connection_id
    ) == ([], [])


@pytest.mark.usefixtures("junction_sources")
@pytest.mark.parametrize(
    "corruption", [{"table_id": "hub-b"}, {"source_column": "wrong"}]
)
def test_canonical_owner_rejects_semantic_but_preserves_declared_path(
    corruption: dict[str, str],
) -> None:
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(**corruption)
    )
    junctions, paths = attributes.find_connected_junction_tables(
        ["hub-a"], connection_id="a"
    )
    assert junctions == [{"id": "left", "name": "left"}]
    assert len(paths) == 1 and paths[0][0]["source_table"] == "left"
    # Administrative inspection retains its unscoped legacy graph behavior.
    assert len(attributes.find_connected_junction_tables(["hub-a"])[0]) == 2


@pytest.mark.usefixtures("junction_sources")
@pytest.mark.parametrize("edge_kind", ["declared", "semantic"])
def test_foreign_junction_cannot_connect_to_local_target(edge_kind: str) -> None:
    if edge_kind == "declared":
        store().query_write(
            s.column__foreign_key.insert().values(
                source_column_id="c-hub-b", target_column_id="c-hub-a"
            )
        )
    else:
        store().query_write(
            s.column__semantic_fk.insert().values(
                column_id="c-hub-b", attribute_id="attr-a", use_in_join_paths=True
            )
        )
    junctions, _ = attributes.find_connected_junction_tables(
        ["hub-a", "hub-b"], connection_id="a"
    )
    assert {row["id"] for row in junctions} == {"left", "right"}


@pytest.mark.usefixtures("junction_sources")
def test_parallel_edges_deduplicate_and_self_disabled_edges_are_excluded() -> None:
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id="c-left", attribute_id="attr-a", use_in_join_paths=True
        )
    )
    junctions, paths = attributes.find_connected_junction_tables(
        ["hub-a", "hub-b"], connection_id="a"
    )
    assert len(junctions) == len(paths) == 2
    store().query_write(
        s.column__semantic_fk.update()
        .where(s.column__semantic_fk.c.column_id == "c-right")
        .values(use_in_join_paths=False)
    )
    assert attributes.find_connected_junction_tables(["hub-a"], connection_id="a")[
        0
    ] == [{"id": "left", "name": "left"}]
    assert attributes.find_connected_junction_tables(["hub-b"]) == ([], [])


def test_empty_input_and_empty_source_do_not_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_store() -> None:
        pytest.fail("empty input or connection ID must not query metadata")

    monkeypatch.setattr(attributes, "store", unexpected_store)
    assert attributes.find_connected_junction_tables([], connection_id="a") == ([], [])
    assert attributes.find_connected_junction_tables(["hub-a"], connection_id="") == (
        [],
        [],
    )
    assert not attributes.column_participates_in_semantic_fk(
        "c-right", connection_id=""
    )
    assert attributes.find_anchor_hub_siblings("left", connection_id="") == ([], 0)


@pytest.mark.usefixtures("junction_sources")
def test_participation_requires_both_endpoints_in_selected_source() -> None:
    for column in ("c-right", "c-hub-a"):
        assert attributes.column_participates_in_semantic_fk(column, connection_id="a")
        for cid in ("b", "unknown", ""):
            assert not attributes.column_participates_in_semantic_fk(
                column, connection_id=cid
            )
    store().query_write(
        s.column__semantic_fk.insert().values(column_id="c-left", attribute_id="attr-b")
    )
    assert not attributes.column_participates_in_semantic_fk(
        "c-left", connection_id="a"
    )
    assert attributes.column_participates_in_semantic_fk("c-left")
    store().query_write(
        s.column__semantic_fk.update()
        .where(s.column__semantic_fk.c.column_id == "c-right")
        .values(use_in_join_paths=False)
    )
    assert attributes.column_participates_in_semantic_fk("c-right", connection_id="a")
    assert not attributes.column_participates_in_semantic_fk(
        "missing", connection_id="a"
    )


@pytest.mark.usefixtures("junction_sources")
@pytest.mark.parametrize(
    "corruption", [{"table_id": "hub-b"}, {"source_column": "wrong"}]
)
def test_participation_requires_canonical_owner_on_both_sides(
    corruption: dict[str, str],
) -> None:
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(**corruption)
    )
    for column in ("c-right", "c-hub-a"):
        assert not attributes.column_participates_in_semantic_fk(
            column, connection_id="a"
        )
        assert attributes.column_participates_in_semantic_fk(column)


@pytest.mark.usefixtures("junction_sources")
def test_foreign_incoming_edge_does_not_make_local_target_participate() -> None:
    store().query_write(s.column__semantic_fk.delete())
    store().query_write(
        s.column__semantic_fk.insert().values(
            column_id="c-hub-b", attribute_id="attr-a"
        )
    )
    assert not attributes.column_participates_in_semantic_fk(
        "c-hub-a", connection_id="a"
    )
    assert attributes.column_participates_in_semantic_fk("c-hub-a")


@pytest.fixture
def sibling_sources(junction_sources: None) -> None:
    for column in ("c-left", "c-hub-b"):
        store().query_write(
            s.column__semantic_fk.insert().values(
                column_id=column, attribute_id="attr-a", use_in_join_paths=True
            )
        )


@pytest.mark.usefixtures("sibling_sources")
def test_siblings_filter_all_sources_before_cap_and_keep_admin_behavior() -> None:
    results, truncated = attributes.find_anchor_hub_siblings(
        "left", max_siblings=None, connection_id="a"
    )
    assert {row["id"] for row in results} == {"hub-a", "right"} and truncated == 0
    assert next(row for row in results if row["id"] == "hub-a")["is_hub"]
    assert all(row["source_table"] == "left" for row in results)
    capped, truncated = attributes.find_anchor_hub_siblings(
        "left", max_siblings=0, connection_id="a"
    )
    assert [row["id"] for row in capped] == ["hub-a"] and truncated == 1
    assert attributes.find_anchor_hub_siblings(
        "left", max_siblings=1, connection_id="a"
    ) == (results, 0)
    assert {
        row["id"] for row in attributes.find_anchor_hub_siblings("left", None)[0]
    } == {"hub-a", "hub-b", "right"}
    for cid in ("b", "unknown", ""):
        assert attributes.find_anchor_hub_siblings("left", connection_id=cid) == ([], 0)
    assert attributes.find_anchor_hub_siblings("hub-b", connection_id="a") == ([], 0)


@pytest.mark.usefixtures("sibling_sources")
@pytest.mark.parametrize(
    "corruption", [{"table_id": "hub-b"}, {"source_column": "wrong"}]
)
def test_siblings_require_canonical_hub_owner(corruption: dict[str, str]) -> None:
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(**corruption)
    )
    assert attributes.find_anchor_hub_siblings("left", connection_id="a") == ([], 0)
    assert attributes.find_anchor_hub_siblings("left")[0]


@pytest.mark.usefixtures("sibling_sources")
def test_siblings_reject_foreign_hub_even_with_valid_canonical_owner() -> None:
    store().query_write(
        s.column__has_attribute.update()
        .where(s.column__has_attribute.c.attribute_id == "attr-a")
        .values(column_id="c-hub-b")
    )
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(table_id="hub-b")
    )
    assert attributes.find_anchor_hub_siblings("left", connection_id="a") == ([], 0)
    assert attributes.find_anchor_hub_siblings("left")[0]


@pytest.mark.usefixtures("sibling_sources")
def test_siblings_keep_pk_and_selected_edge_constraints() -> None:
    store().query_write(
        s.column__semantic_fk.update()
        .where(s.column__semantic_fk.c.column_id == "c-right")
        .values(use_in_join_paths=False)
    )
    assert attributes.find_anchor_hub_siblings("left", connection_id="a") == ([], 0)
    store().query_write(s.column__semantic_fk.update().values(use_in_join_paths=True))
    store().query_write(
        s.catalog_table.update()
        .where(s.catalog_table.c.id == "hub-a")
        .values(pk=["not_id"])
    )
    assert attributes.find_anchor_hub_siblings("left", connection_id="a") == ([], 0)
