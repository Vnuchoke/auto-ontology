# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Destructive resets against isolated migrated Postgres and real pgvector."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Table, null, select

from auto_ontology.dal import reset as r, schema as s
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)
from auto_ontology.dal.tests.test_reset import _add, _exists, _link
from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import write_embedded_rows


@pytest.fixture
def sources(identity_database: None) -> Iterator[tuple[str, str]]:
    from alembic import command
    from alembic.config import Config

    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    ids = tuple(
        _add(
            s.catalog_database,
            id=cid,
            name="sales",
            display_label=label,
            configuration_origin="managed",
            credential_ref=f"dummy/{cid}",
            connection={"database": "sales", "password": "dummy"},
            description="curated root",
        )
        for cid, label in (("Opaque-A", "First sales"), ("opaque-a", "Second sales"))
    )
    try:
        yield ids
    finally:
        dispose_engine()


def snapshot(table: Table, **filters: Any) -> list[dict[str, Any]]:
    return store().query_read(select(table).filter_by(**filters))


def catalog(cid: str) -> tuple[str, str]:
    schema = _add(s.catalog_schema, database_id=cid, name="public")
    table = _add(s.catalog_table, schema_id=schema, name="orders")
    _add(s.catalog_column, table_id=table, name="id")
    return schema, table


def vector_client(collection: str, **kwargs: Any) -> PostgresVDB:
    return PostgresVDB(
        connection_string=get_postgres_connection_string(),
        schema_name="vdb",
        collection_name=collection,
        **kwargs,
    )


def vector(cid: str | None, label: str = "Term") -> dict[str, Any]:
    return {
        "text": f"{cid} {label}",
        "metadata": {
            "id": "shared",
            "label": label,
            "embedding": [1.0, 0.0, 0.0],
            **({"connection_id": cid} if cid is not None else {}),
        },
    }


@pytest.mark.parametrize("reset", [r.delete_all_data, r.delete_semantic_layer])
def test_reset_uses_ownership_and_preserves_other_source(
    sources: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    reset: Callable[[str], Any],
) -> None:
    a, b = sources
    schema_a, table_a = catalog(a)
    _, table_b = catalog(b)
    roots = snapshot(s.catalog_database)
    before_b = snapshot(s.catalog_table, id=table_b)
    shared = _add(s.term, name="Shared")
    _link(s.table__term, table_id=table_a, term_id=shared)
    attr_b = _add(s.sql_attribute, name="constant", connection_id=b)
    _link(s.sql_attribute__term, attribute_id=attr_b, term_id=shared)
    # No represented table B: the SQL-attribute link alone must protect the Term.
    attr_a = _add(s.sql_attribute, name="constant", connection_id=a)
    custom_a = _add(s.custom_analysis, name="constant", connection_id=a)
    query_a = _add(s.sql_query, connection_id=a, sql_full_query="SELECT 1")
    query_b = _add(s.sql_query, connection_id=b, sql_full_query="SELECT 1")
    legacy = _add(s.sql_query, sql_full_query="SELECT 1")
    _link(s.sql_query__table, sql_query_id=legacy, table_id=table_a)
    unknown = _add(s.sql_attribute, name="legacy")
    _link(s.sql_attribute__sql, attribute_id=unknown, sql_query_id=legacy)
    pql = _add(s.pql_analysis, name="global", pql="PREDICT x")
    data, semantic, global_pql = (
        vector_client("data_layer"),
        vector_client("semantic_layer"),
        vector_client("semantic_layer", global_pql=True),
    )
    try:
        for client in (data, semantic):
            write_embedded_rows(client, [vector(a), vector(b)])
        b_vectors = [
            client.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": b})
            for client in (data, semantic)
        ]
        write_embedded_rows(global_pql, [vector(None, "PqlAnalysis")])
        monkeypatch.setattr(r, "get_data_vdb", lambda: vector_client("data_layer"))
        monkeypatch.setattr(
            r, "get_semantic_vdb", lambda: vector_client("semantic_layer")
        )
        reset(a)
        assert snapshot(s.catalog_database) == roots
        assert snapshot(s.catalog_table, id=table_b) == before_b
        assert [
            client.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": b})
            for client in (data, semantic)
        ] == b_vectors
        for table, row in (
            (s.term, shared),
            (s.sql_attribute, attr_b),
            (s.sql_query, query_b),
            (s.sql_query, legacy),
            (s.sql_attribute, unknown),
            (s.pql_analysis, pql),
        ):
            assert _exists(table, row)
        assert snapshot(s.sql_attribute__term, attribute_id=attr_b)
        for table, row in (
            (s.sql_attribute, attr_a),
            (s.custom_analysis, custom_a),
            (s.sql_query, query_a),
        ):
            assert not _exists(table, row)
        assert not snapshot(s.table__term, table_id=table_a)
        assert _exists(s.catalog_schema, schema_a) == (reset is r.delete_semantic_layer)
        assert (
            len(semantic.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": b})[0])
            == 1
        )
        assert semantic.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": a}) == [[]]
        assert len(global_pql.retrieval([[1.0, 0.0, 0.0]])[0]) == 1
        r.delete_all_data()
        assert snapshot(s.catalog_database) == roots
        assert _exists(s.pql_analysis, pql)
        assert len(global_pql.retrieval([[1.0, 0.0, 0.0]])[0]) == 1
        # Re-ingest uses the same reserved identity/label.
        catalog(a)
        assert snapshot(s.catalog_database) == roots
    finally:
        for client in (data, semantic, global_pql):
            client.close()


def test_final_remove_refuses_active_and_retry_never_resolves_reused_label(
    sources: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = sources
    catalog(a)
    before = snapshot(s.catalog_database)
    with pytest.raises(ValueError, match="retired"):
        r.remove_connection_data(a)
    assert snapshot(s.catalog_database) == before

    class Vectors:
        fail = True
        closed = 0
        deleted: list[str] = []

        def delete_by_connection(self, cid: str) -> list[str]:
            if self.fail:
                raise RuntimeError("vector unavailable")
            self.deleted.append(cid)
            return []

        def close(self) -> None:
            self.closed += 1

    vectors = Vectors()
    monkeypatch.setattr(r, "get_semantic_vdb", lambda: vectors)
    monkeypatch.setattr(r, "get_data_vdb", lambda: vectors)
    store().query_write(
        s.catalog_database.update()
        .where(s.catalog_database.c.id == a)
        .values(configuration_origin=None, credential_ref=None, connection=null())
    )
    with pytest.raises(RuntimeError, match="vector unavailable"):
        r.remove_connection_data(a)
    assert vectors.closed == 1
    assert _exists(s.catalog_database, a), "failed vectors must retain reserved label"
    vectors.fail = False
    r.remove_connection_data(a)
    assert not _exists(s.catalog_database, a)
    replacement = _add(
        s.catalog_database,
        name="sales",
        display_label="First sales",
        configuration_origin="env",
    )
    catalog(replacement)
    roots = snapshot(s.catalog_database)
    children = snapshot(s.catalog_schema)
    r.remove_connection_data(a)
    assert snapshot(s.catalog_database) == roots
    assert snapshot(s.catalog_schema) == children
    assert _exists(s.catalog_database, b)
    assert vectors.deleted == [a, a, a, a]


def test_physical_names_and_labels_are_not_id_fallbacks(
    sources: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal.tests.test_reset import _StubVdb

    a, b = sources
    catalog(a)
    catalog(b)
    roots, children = snapshot(s.catalog_database), snapshot(s.catalog_schema)
    monkeypatch.setattr(r, "get_semantic_vdb", _StubVdb)
    monkeypatch.setattr(r, "get_data_vdb", _StubVdb)
    r.delete_all_data("sales")
    r.delete_all_data("First sales")
    assert snapshot(s.catalog_database) == roots
    assert snapshot(s.catalog_schema) == children


def test_semantic_reset_preserves_terms_needed_by_column_attrs_and_global_tags(
    sources: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal.tests.test_reset import _StubVdb

    a, b = sources
    _, table_a = catalog(a)
    _, table_b = catalog(b)
    protected = []
    for name in ("column-needed", "tag-needed", "exclusive"):
        term = _add(s.term, name=name)
        protected.append(term)
        _link(s.table__term, table_id=table_a, term_id=term)
    attr_b = _add(
        s.column_attribute,
        table_id=table_b,
        name="id",
        source_column="id",
        term_name="column-needed",
    )
    _link(s.column_attribute__term, attribute_id=attr_b, term_id=protected[0])
    tag = _add(s.tag, name="curated")
    _link(s.tag_target, tag_id=tag, term_id=protected[1])
    attr_a = _add(
        s.column_attribute,
        table_id=table_a,
        name="id",
        source_column="id",
        term_name="exclusive",
    )
    _link(s.column_attribute__term, attribute_id=attr_a, term_id=protected[2])
    monkeypatch.setattr(r, "get_semantic_vdb", _StubVdb)
    r.delete_semantic_layer(a)
    assert _exists(s.column_attribute, attr_b)
    assert not _exists(s.column_attribute, attr_a)
    assert _exists(s.term, protected[0])
    assert _exists(s.term, protected[1])
    assert not _exists(s.term, protected[2])
    assert snapshot(s.column_attribute__term, attribute_id=attr_b)
    assert snapshot(s.tag_target, tag_id=tag)


@pytest.mark.parametrize(
    "configured",
    [
        {"configuration_origin": "env"},
        {"credential_ref": "dummy/ref"},
        {"connection": {}},
        {"connection": None},  # JSON null is not SQL NULL: still not retired.
    ],
)
def test_final_remove_checks_every_configuration_field_before_cleanup(
    sources: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    configured: dict[str, Any],
) -> None:
    a, _ = sources
    catalog(a)
    store().query_write(
        s.catalog_database.update()
        .where(s.catalog_database.c.id == a)
        .values(configuration_origin=None, credential_ref=None, connection=null())
    )
    store().query_write(
        s.catalog_database.update()
        .where(s.catalog_database.c.id == a)
        .values(**configured)
    )
    before = snapshot(s.catalog_database)
    children = snapshot(s.catalog_schema)
    monkeypatch.setattr(
        r, "get_semantic_vdb", lambda: pytest.fail("cleanup before retirement")
    )
    with pytest.raises(ValueError, match="retired"):
        r.remove_connection_data(a)
    assert snapshot(s.catalog_database) == before
    assert snapshot(s.catalog_schema) == children
