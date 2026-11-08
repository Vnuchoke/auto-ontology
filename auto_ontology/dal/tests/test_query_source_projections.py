# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.dal import datasources, schema as s, terms
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_term_projections import (
    identity_database as identity_database,
    projections as projections,
)


@pytest.mark.usefixtures("projections")
def test_readiness_is_source_projection_not_global_vocabulary() -> None:
    store().query_write(s.table__term.delete().where(s.table__term.c.table_id == "t-b"))
    store().query_write(
        s.column_attribute__term.delete().where(
            s.column_attribute__term.c.attribute_id == "attr-b"
        )
    )
    assert terms.semantic_layer_calculated()
    assert terms.semantic_layer_calculated(connection_id="a")
    assert not terms.semantic_layer_calculated(connection_id="b")
    assert not terms.semantic_layer_calculated(connection_id="")


@pytest.mark.usefixtures("projections")
def test_table_context_filters_foreign_ids_before_samples() -> None:
    rows = datasources.fetch_tables_by_ids(["t-a", "t-b"], connection_id="a")
    assert [r["id"] for r in rows] == ["t-a"]
    assert rows[0]["connection_id"] == "a"
    assert "a_marker" in str(rows)
    assert "b_marker" not in str(rows)
    columns = datasources.fetch_col_table_contexts(["c-a", "c-b"], connection_id="a")
    assert list(columns) == ["c-a"]
    assert columns["c-a"]["connection_id"] == "a"


@pytest.mark.usefixtures("projections")
def test_candidate_enrichment_validates_relation_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.retrieval.data_access import candidates

    monkeypatch.setattr(
        candidates,
        "search_semantic_index",
        lambda *a, **kw: [
            {"id": "c-a", "label": "Column", "connection_id": "a", "text": "A"},
            # A compromised retriever claims the selected CID for a foreign node ID.
            {"id": "c-b", "label": "Column", "connection_id": "a", "text": "B"},
        ],
    )
    rows = candidates._get_candidates_information(None, "status", connection_id="a")
    assert [r["id"] for r in rows] == ["c-a"]
    assert "a_marker" in str(rows)
    assert "b_marker" not in str(rows)


@pytest.mark.usefixtures("projections")
def test_coverage_rejects_foreign_relation_ids_even_with_forged_hit_cid() -> None:
    from auto_ontology.retrieval.entity_coverage.agents.coverage import (
        CoverageGradeAgent,
    )

    state = {
        "connection_id": "a",
        "path_state": {
            "connection_id": "a",
            "target_db": "sales",
            "entities": ["A", "B"],
            "return_uncovered_entities": True,
            "retrieved_column_attributes": [
                {
                    "id": "attr-a",
                    "connection_id": "a",
                    "score": 0.1,
                    "query_entity": "A",
                },
                {
                    "id": "attr-b",
                    "connection_id": "a",
                    "score": 0.1,
                    "query_entity": "B",
                },
            ],
        },
    }
    response = CoverageGradeAgent().execute(state)["path_state"]["final_response"]
    assert response["coverage"] == 0.5
    assert [r["id"] for r in response["candidates"]] == ["attr-a"]
    assert response["uncovered_entities"] == ["B"]


@pytest.mark.usefixtures("projections")
def test_term_reads_project_shared_vocabulary_before_context_hydration() -> None:
    rows = terms.fetch_all_terms(connection_id="a")
    assert [r["id"] for r in rows] == ["shared"]
    assert rows[0]["connection_id"] == "a"
    assert rows[0]["schema_names"] == ["schema_a"]
    full = terms.get_full_term_by_id("shared", connection_id="a")
    assert [t["id"] for t in full["tables"]] == ["t-a"]
    attrs = terms.fetch_column_attributes_by_term_id("shared", connection_id="a")
    assert [a["id"] for a in attrs] == ["attr-a"]
    assert "a_marker" in str(attrs) and "b_marker" not in str(attrs)


@pytest.mark.usefixtures("projections")
def test_catalog_read_boundaries_reject_foreign_ids_and_ambiguous_names() -> None:
    assert datasources.find_table_id_by_name("orders", "sales") is None
    assert (
        datasources.find_table_id_by_name("orders", "sales", connection_id="a") == "t-a"
    )
    assert (
        datasources.find_column_id_by_table_and_name("orders", "status", "sales")
        is None
    )
    assert (
        datasources.find_column_id_by_table_and_name(
            "orders", "status", "sales", connection_id="a"
        )
        == "c-a"
    )
    assert datasources.find_table_key_columns("orders", "sales", connection_id="a") == {
        "pk": [],
        "unique": [],
    }
    assert datasources.fetch_table_by_id("t-b", connection_id="a") is None
    assert datasources.fetch_schemas_for_database("b", connection_id="a") is None
    assert datasources.fetch_tables_for_schema("s-b", connection_id="a") == []
    assert datasources.fetch_columns_for_table("t-b", connection_id="a") is None
    assert datasources.count_columns_for_table("t-b", connection_id="a") == 0
    assert (
        datasources.fetch_parent_table_id_for_column("c-b", connection_id="a") is None
    )
    assert datasources.fetch_join_neighbors("t-b", connection_id="a") == []
    assert [r["id"] for r in datasources.fetch_databases(connection_id="a")] == ["a"]
    assert terms.fetch_terms_and_attributes_for_table("t-b", connection_id="a") == (
        [],
        [],
    )
    assert terms.count_terms(connection_id="a") == 1


@pytest.mark.usefixtures("projections")
def test_foreign_attribute_description_never_enters_table_context() -> None:
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-b")
        .values(description="b_secret_description")
    )
    store().query_write(
        s.column__semantic_fk.insert().values(column_id="c-a", attribute_id="attr-b")
    )
    rows = datasources.fetch_tables_by_ids(["t-a"], connection_id="a")
    assert "b_secret_description" not in str(rows)


@pytest.mark.usefixtures("projections")
def test_attribute_only_term_is_queryable_without_representing_table() -> None:
    store().query_write(s.table__term.delete())
    assert terms.count_terms(connection_id="a") == 1
    assert [r["id"] for r in terms.fetch_all_terms(connection_id="a")] == ["shared"]


@pytest.mark.usefixtures("projections")
@pytest.mark.parametrize("local_description", [None, "a_local_description"])
@pytest.mark.parametrize("node_id", ["c-a", "t-a"])
def test_catalog_refresh_scopes_descriptions_before_embedding_text(
    monkeypatch: pytest.MonkeyPatch, local_description: str | None, node_id: str
) -> None:
    from unittest.mock import Mock

    from auto_ontology.server.datasources import service
    from auto_ontology.utils import embedding
    import auto_ontology.utils as utils
    import auto_ontology.vdb as vdb
    from auto_ontology.vdb import records

    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-a")
        .values(description=local_description)
    )
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "attr-b")
        .values(description="b_secret_description")
    )
    store().query_write(
        s.column__semantic_fk.insert().values(column_id="c-a", attribute_id="attr-b")
    )
    captured: list[dict] = []
    monkeypatch.setattr(utils, "get_embed_params", lambda: None)
    monkeypatch.setattr(
        embedding,
        "embed_rows",
        lambda rows, *_args, **_kwargs: captured.extend(rows) or rows,
    )
    monkeypatch.setattr(vdb, "get_data_vdb", Mock(return_value=Mock()))
    monkeypatch.setattr(records, "write_embedded_rows", lambda _vdb, rows: len(rows))

    service._refresh_vdb_embeddings([node_id])
    assert captured
    assert {r["metadata"]["connection_id"] for r in captured} == {"a"}
    model_text = " ".join(r["text"] for r in captured)
    assert "b_secret_description" not in model_text
    assert "b_marker" not in model_text
    if node_id == "c-a":
        assert "a_marker" in model_text
    if local_description:
        assert local_description in model_text

    tables, columns, database_name = datasources.fetch_tables_and_columns_by_node_ids(
        [node_id]
    )
    assert database_name == "sales"
    assert tables["id"].tolist() == ["t-a"]
    assert columns["id"].tolist() == ["c-a"]
    assert columns["connection_id"].tolist() == ["a"]
    assert "a_marker" in str(columns.iloc[0]["sample_values"])
    assert "b_secret_description" not in str(columns.to_dict("records"))
    assert columns.iloc[0]["description"] == local_description


@pytest.mark.usefixtures("projections")
def test_catalog_refresh_source_selection_rejects_mixed_and_keeps_unknown_empty() -> (
    None
):
    with pytest.raises(ValueError, match="single connection_id"):
        datasources.fetch_tables_and_columns_by_node_ids(["t-a", "c-b"])
    tables, columns, name = datasources.fetch_tables_and_columns_by_node_ids(
        ["unknown"]
    )
    assert tables.empty and columns.empty and name == ""
