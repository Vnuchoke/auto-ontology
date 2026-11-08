# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.dal import attributes, datasources, schema as s, terms
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)


@pytest.fixture
def join_sources(identity_database: None) -> Iterator[None]:
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        for cid in ("a", "b"):
            store().query_write(
                s.catalog_database.insert().values(
                    id=cid, name="sales", display_label=cid
                )
            )
            store().query_write(
                s.catalog_schema.insert().values(
                    id=f"s-{cid}", database_id=cid, name="public"
                )
            )
        for tid, cid, name in (
            ("left", "a", "left"),
            ("right", "a", "right"),
            ("hub-a", "a", "orders"),
            ("hub-b", "b", "orders"),
        ):
            store().query_write(
                s.catalog_table.insert().values(
                    id=tid,
                    schema_id=f"s-{cid}",
                    name=name,
                    pk=["id"],
                    description=f"{cid}_description",
                )
            )
            store().query_write(
                s.catalog_column.insert().values(
                    id=f"c-{tid}",
                    table_id=tid,
                    name="id",
                    sample_values=f'["{cid}_sample"]',
                )
            )
        for cid in ("a", "b"):
            store().query_write(
                s.term.insert().values(
                    id=f"term-{cid}",
                    name=f"Term {cid}",
                    source="semantic",
                    synonyms=[f"{cid}_synonym"],
                )
            )
            store().query_write(
                s.column_attribute.insert().values(
                    id=f"attr-{cid}",
                    table_id=f"hub-{cid}",
                    name="Identity",
                    term_name=f"Term {cid}",
                    source_column="id",
                    source="semantic",
                )
            )
            store().query_write(
                s.column_attribute__term.insert().values(
                    attribute_id=f"attr-{cid}", term_id=f"term-{cid}"
                )
            )
            store().query_write(
                s.column__has_attribute.insert().values(
                    attribute_id=f"attr-{cid}", column_id=f"c-hub-{cid}"
                )
            )
        for column in ("c-left", "c-right"):
            store().query_write(
                s.column__semantic_fk.insert().values(
                    column_id=column, attribute_id="attr-b", use_in_join_paths=True
                )
            )
        yield
    finally:
        dispose_engine()


def _assert_no_routes(connection_id: str) -> None:
    assert (
        attributes.find_join_path("c-left", "c-right", connection_id=connection_id)
        == []
    )
    assert (
        attributes.find_shared_hub_bridge(
            "c-left", "c-right", connection_id=connection_id
        )
        == {}
    )
    assert attributes.find_table_bridge(
        "left", "right", connection_id=connection_id
    ) == ([], [])
    assert attributes.find_kept_table_bridges(
        ["left", "right"], connection_id=connection_id
    ) == ([], [], 0)


@pytest.mark.usefixtures("join_sources")
@pytest.mark.parametrize("connection_id", ["a", "b", "unknown", ""])
def test_foreign_intermediate_and_wrong_source_cannot_bridge(
    connection_id: str,
) -> None:
    _assert_no_routes(connection_id)


@pytest.mark.usefixtures("join_sources")
def test_local_routes_work_but_foreign_canonical_owner_is_rejected() -> None:
    store().query_write(s.column__semantic_fk.delete())
    for column in ("c-left", "c-right"):
        store().query_write(
            s.column__semantic_fk.insert().values(
                column_id=column, attribute_id="attr-a", use_in_join_paths=True
            )
        )
    assert len(attributes.find_join_path("c-left", "c-right", connection_id="a")) == 2
    assert attributes.find_shared_hub_bridge(
        "c-left", "c-right", connection_id="a"
    ) == {"hub_table": "orders", "hub_column": "id"}
    bridges, hops = attributes.find_table_bridge("left", "right", connection_id="a")
    assert bridges == [{"id": "hub-a", "name": "orders"}] and len(hops) == 2
    assert attributes.find_kept_table_bridges(["left", "right"], connection_id="a") == (
        bridges,
        [hops],
        0,
    )
    assert attributes.find_table_bridge("left", "right", set(), connection_id="a") == (
        [],
        [],
    )
    assert attributes.find_kept_table_bridges(
        ["left", "right"], max_bridge_tables=0, connection_id="a"
    ) == ([], [], 1)
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(table_id="hub-b")
    )
    _assert_no_routes("a")
    # Explicit administrative inspection retains the legacy graph behavior.
    assert attributes.find_shared_hub_bridge("c-left", "c-right") == {
        "hub_table": "orders",
        "hub_column": "id",
    }


@pytest.mark.usefixtures("join_sources")
def test_foreign_endpoint_ids_are_not_hydrated() -> None:
    assert attributes.find_join_path("c-left", "c-hub-b", connection_id="a") == []
    assert (
        attributes.find_shared_hub_bridge("c-left", "c-hub-b", connection_id="a") == {}
    )
    assert attributes.find_table_bridge("left", "hub-b", connection_id="a") == ([], [])
    assert attributes.find_kept_table_bridges(["left", "hub-b"], connection_id="a") == (
        [],
        [],
        0,
    )


@pytest.mark.usefixtures("join_sources")
def test_synonyms_and_catalog_names_use_persisted_source_ownership() -> None:
    assert terms.fetch_term_synonyms(["attr-a", "attr-b"], connection_id="a") == {
        "Term a": ["a_synonym"]
    }
    assert terms.fetch_term_synonyms(["attr-b"], connection_id="a") == {}
    assert len(terms.fetch_term_synonyms(["attr-a", "attr-b"])) == 2
    assert datasources.fetch_all_schema_ids(connection_id="a") == ["s-a"]
    assert datasources.fetch_all_schema_ids(connection_id="unknown") == []
    assert datasources.fetch_table_by_name("orders", connection_id="a")["id"] == "hub-a"
    assert datasources.fetch_table_by_name("orders", connection_id="b")["id"] == "hub-b"
    assert datasources.fetch_table_by_name("orders") is None
    assert datasources.fetch_table_by_name("orders", connection_id="unknown") is None
    assert terms.fetch_term_synonyms(["attr-a"], connection_id="") == {}


@pytest.mark.usefixtures("join_sources")
def test_relevant_candidate_tables_ignore_forged_identity_and_text() -> None:
    from auto_ontology.retrieval.data_access.relevant_tables import (
        get_relevant_tables_from_candidates,
    )

    candidates = [
        {
            "relevant_tables": [
                {"id": "hub-b", "connection_id": "a", "text": "b_secret"},
                {
                    "id": "hub-a",
                    "connection_id": "a",
                    "description": "b_secret",
                    "text": "b_secret",
                },
            ]
        }
    ]
    result = get_relevant_tables_from_candidates(candidates, connection_id="a")
    assert [row["id"] for row in result] == ["hub-a"]
    assert result[0]["connection_id"] == "a" and result[0]["database_name"] == "sales"
    assert "a_sample" in str(result) and "a_description" in str(result)
    assert "b_secret" not in str(result) and "b_sample" not in str(result)
    assert "relevant_tables" not in candidates[0]


@pytest.mark.usefixtures("join_sources")
@pytest.mark.parametrize(
    "corruption", [{"table_id": "hub-b"}, {"source_column": "not_id"}]
)
def test_candidate_fk_prompt_requires_canonical_attribute_owner(
    corruption: dict[str, str],
) -> None:
    from auto_ontology.retrieval.data_access.relevant_tables import (
        get_relevant_tables_from_candidates,
    )
    from auto_ontology.retrieval.text_to_sql.formatters_util import (
        format_tables_for_prompt,
    )

    store().query_write(s.column__semantic_fk.delete())
    for col in ("c-left", "c-right", "c-hub-a"):
        store().query_write(
            s.column__semantic_fk.insert().values(column_id=col, attribute_id="attr-a")
        )
    tables = get_relevant_tables_from_candidates(
        [{"relevant_tables": [{"id": tid} for tid in ("left", "right", "hub-a")]}],
        connection_id="a",
    )
    assert len(tables) == 3
    assert all(t["columns"][0]["semantic_fk"] == "public.orders.id" for t in tables)
    assert "public.orders.id" in format_tables_for_prompt(tables)

    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(
            **corruption,
            description="b_canonical_description"
            if "table_id" in corruption
            else "a_description",
        )
    )
    # HAS_ATTRIBUTE still points to c-hub-a, whose ancestry remains A.
    tables = get_relevant_tables_from_candidates(
        [{"relevant_tables": [{"id": tid} for tid in ("left", "right")]}],
        connection_id="a",
    )
    assert {t["connection_id"] for t in tables} == {"a"}
    prompt = format_tables_for_prompt(tables)
    assert "public.orders.id" not in prompt
    assert all(t["columns"][0]["semantic_fk"] is None for t in tables)
    assert "b_canonical_description" not in prompt
    assert "b_sample" not in prompt
    admin = datasources.fetch_tables_by_ids(["left", "right"])
    assert all(t["columns"][0]["semantic_fk"] == "public.orders.id" for t in admin)

    # An independent valid parallel target must survive the invalid relation.
    store().query_write(
        s.column_attribute.insert().values(
            id="valid-alt",
            table_id="hub-a",
            source_column="id",
            name="Alternate identity",
            term_name="Term a",
            source="semantic",
        )
    )
    store().query_write(
        s.column__has_attribute.insert().values(
            attribute_id="valid-alt", column_id="c-hub-a"
        )
    )
    store().query_write(
        s.column__semantic_fk.insert().values(
            attribute_id="valid-alt", column_id="c-right"
        )
    )
    tables = datasources.fetch_tables_by_ids(["left", "right"], connection_id="a")
    assert tables[0]["columns"][0]["semantic_fk"] is None
    assert tables[1]["columns"][0]["semantic_fk"] == "public.orders.id"
