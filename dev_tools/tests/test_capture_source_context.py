# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture plans must select existing fixture catalog IDs, never credentials."""

from typing import Any
from unittest.mock import Mock

import pytest

from dev_tools.fixtures import capture_dal_golden as capture


def test_capture_plan_passes_source_ids_to_projection_and_query_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = dict.fromkeys(
        (
            "table_film table_rental term_Film term_Customer attr_film_id "
            "sqlattr_revenue analysis_top_films db_pagila db_chinook schema_public "
            "col_film_title col_rental_customer_id col_film_film_id "
            "col_customer_customer_id col_inventory_film_id pql_churn "
            "zone_film zone_retired"
        ).split(),
        "fixture-entity",
    )
    ids.update(db_pagila="catalog-pagila", db_chinook="catalog-chinook")
    monkeypatch.setattr(capture, "_fixture_ids", lambda: ids)
    monkeypatch.setattr(capture, "build_id_map", lambda: {})
    targets = (
        "terms.fetch_all_terms_and_attributes",
        "terms.fetch_term_and_column_attributes_for_embedding",
        "sql_attributes.find_attr_by_name",
        "custom_analyses.find_analysis_by_name",
        "custom_analyses.fetch_custom_analyses",
        "sql_attributes.fetch_sql_attributes_with_sql",
        "sql_attributes.fetch_sql_attribute_docs",
        "sql_attributes.fetch_tables_from_sql_attributes",
        "custom_analyses.fetch_custom_analyses_with_sql",
        "custom_analyses.fetch_tables_from_custom_analyses",
        "datasources.fetch_bridge_table_candidates",
        "terms.fetch_table_schema_map",
        "terms.fetch_terms_with_sqls",
        "attributes.fetch_attr_column_contexts",
    )
    spies: dict[str, Mock] = {}
    for target in targets:

        def scoped(*args: Any, connection_id: str) -> list[Any]:
            assert connection_id in {"catalog-pagila", "catalog-chinook"}
            return []

        spies[target] = Mock(side_effect=scoped)
        monkeypatch.setattr(f"auto_ontology.dal.{target}", spies[target])

    original_run = capture.Capture.run

    def run_selected(self: capture.Capture, name: str, fn: Any) -> None:
        if name.split("[")[0].split("__")[0] in targets:
            original_run(self, name, fn)

    monkeypatch.setattr(capture.Capture, "run", run_selected)
    cap = capture.capture_all()

    assert cap.errors == {}
    for target, spy in spies.items():
        expected = (
            ["catalog-pagila", "catalog-chinook"]
            if target == "terms.fetch_all_terms_and_attributes"
            else ["catalog-pagila"] * (2 if target.endswith("find_attr_by_name") else 1)
        )
        assert [
            call.kwargs.get("connection_id") for call in spy.call_args_list
        ] == expected
    assert ("terms", "fetch_all_terms_and_attributes") not in capture.ZONE_SCOPED_READS
    assert ("custom_analyses", "fetch_custom_analyses") not in capture.ZERO_ARG_READS


def test_fixture_roots_use_catalog_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auto_ontology.dal.datasources.fetch_databases",
        lambda: [
            {"id": "catalog-pagila", "name": "pagila"},
            {"id": "catalog-chinook", "name": "chinook"},
        ],
    )
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        Mock(side_effect=AssertionError("catalog-only fixtures need no live source")),
    )
    assert capture._fixture_databases() == {
        "pagila": "catalog-pagila",
        "chinook": "catalog-chinook",
    }


@pytest.mark.parametrize("entrypoint", ["_fixture_ids", "build_id_map"])
@pytest.mark.parametrize("name", ["pagila", "chinook"])
def test_ambiguous_physical_roots_rejected_before_selection_or_normalisation(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, name: str
) -> None:
    for read in ("fetch_schemas_by_ids", "fetch_schemas_for_database"):
        monkeypatch.setattr(
            f"auto_ontology.dal.datasources.{read}",
            Mock(side_effect=AssertionError("ambiguity must fail before other reads")),
        )
    monkeypatch.setattr(
        "auto_ontology.dal.datasources.fetch_databases",
        lambda: [{"id": "first", "name": name}, {"id": "second", "name": name}],
    )
    with pytest.raises(ValueError, match=name):
        getattr(capture, entrypoint)()
