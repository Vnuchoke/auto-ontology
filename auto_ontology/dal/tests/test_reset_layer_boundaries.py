# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.dal import reset, schema as s, sql_attributes, terms
from auto_ontology.dal.tests.test_connection_identity_schema import (
    identity_database as identity_database,
)
from auto_ontology.dal.tests.test_reset import _StubVdb, _add, _exists, _link
from auto_ontology.dal.tests.test_source_scoped_reset import catalog, sources as sources


def test_semantic_reset_preserves_ingestion_history_for_rebuild(
    sources: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _ = sources
    _, table = catalog(a)
    query = _add(s.sql_query, connection_id=a, sql_full_query="SELECT id FROM orders")
    _link(s.sql_query__table, sql_query_id=query, table_id=table)
    monkeypatch.setattr(reset, "get_semantic_vdb", _StubVdb)
    reset.delete_semantic_layer(a)
    assert _exists(s.sql_query, query), "semantic rebuild needs ingested SQL history"
    term = _add(s.term, name="Rebuilt orders", source="semantic")
    _link(s.table__term, table_id=table, term_id=term)
    assert (
        terms.fetch_terms_with_sqls(connection_id=a)[0]["sqls"][0]["props"]["id"]
        == query
    )


def test_data_reset_preserves_sql_of_retained_semantic_owner(
    sources: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, _ = sources
    _, table = catalog(a)
    term = _add(s.term, name="Shared")
    query = _add(s.sql_query, connection_id=a, sql_full_query="SELECT id FROM orders")
    attr = _add(s.sql_attribute, connection_id=a, name="Order identifier")
    _link(s.sql_attribute__term, attribute_id=attr, term_id=term)
    _link(s.sql_attribute__sql, attribute_id=attr, sql_query_id=query)
    _link(s.sql_query__table, sql_query_id=query, table_id=table)
    monkeypatch.setattr(reset, "get_data_vdb", _StubVdb)
    reset.delete_data_layer(a)
    row = sql_attributes.get_full_sql_attribute_by_id(attr)
    assert row is not None and row["sql"] == "SELECT id FROM orders"
