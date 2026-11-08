# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.dal import (
    custom_analyses as ca,
    datasources,
    schema as s,
    sql_attributes as sa,
)
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    identity_database as identity_database,
)
from auto_ontology.dal.tests.test_source_scoped_definitions import (
    definitions as definitions,
)


@pytest.mark.usefixtures("definitions")
def test_docs_filter_source_and_legacy_before_rendering():
    for cid in ("a", "b"):
        docs = ca._custom_analysis_docs(None, connection_id=cid)
        assert [doc["id"] for doc in docs] == [f"custom_analysis-{cid}"]
        assert docs[0]["connection_id"] == cid
        for suffix in ("a", "b", "legacy"):
            docs = sa.fetch_sql_attribute_docs(
                f"sql_attribute-{suffix}", connection_id=cid
            )
            assert bool(docs) == (suffix == cid)
            if docs:
                assert docs[0]["connection_id"] == cid
                assert f"description: {cid} only" in docs[0]["text"]


@pytest.mark.usefixtures("definitions")
@pytest.mark.parametrize(
    "relation", ["table", "column", "statement", "legacy_statement"]
)
def test_any_foreign_relation_rejects_entire_doc(relation):
    if relation in ("table", "column"):
        link = s.sql_query__table if relation == "table" else s.sql_query__column
        key = "table_id" if relation == "table" else "column_id"
        store().query_write(
            link.insert().values(
                sql_query_id="q-a", **{key: "t-b" if relation == "table" else "c-b"}
            )
        )
    else:
        query_id = "q-b" if relation == "statement" else "q-legacy"
        store().query_write(
            s.sql_attribute__sql.insert().values(
                attribute_id="sql_attribute-a", sql_query_id=query_id
            )
        )
        store().query_write(
            s.custom_analysis__sql.insert().values(
                analysis_id="custom_analysis-a", sql_query_id=query_id
            )
        )
    assert sa.fetch_sql_attribute_docs("sql_attribute-a", connection_id="a") == []
    assert ca._custom_analysis_docs("custom_analysis-a", connection_id="a") == []


@pytest.mark.usefixtures("definitions")
def test_refresh_frames_carry_ancestor_identity_and_reject_mixed_ids():
    tables, columns, name = datasources.fetch_tables_and_columns_by_node_ids(["c-a"])
    assert name == "sales"
    assert tables["connection_id"].tolist() == ["a"]
    assert columns["connection_id"].tolist() == ["a"]
    with pytest.raises(ValueError, match="single connection_id"):
        datasources.fetch_tables_and_columns_by_node_ids(["t-a", "c-b"])
