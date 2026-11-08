# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source projections round-trip through the installed pgvector store."""

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import to_vdb_records


def row(cid: str, text: str = "A", vector: list | None = None) -> dict:
    return {
        "text": text,
        "metadata": {
            "id": "shared-term",
            "connection_id": cid,
            "database_name": "sales",
            "label": "Term",
            "embedding": vector or [1.0, 0.0, 0.0],
        },
    }


@pytest.fixture
def store(identity_database: None) -> Iterator[PostgresVDB]:
    from auto_ontology.dal.session import dispose_engine

    dispose_engine()
    vdb = PostgresVDB(
        connection_string=get_postgres_connection_string(),
        collection_name="semantic_layer",
        schema_name="vdb",
    )
    try:
        yield vdb
    finally:
        vdb.close()
        dispose_engine()


def snapshot(store: PostgresVDB) -> list:
    with psycopg.connect(store.connection_string) as conn:
        return conn.execute(
            "SELECT langchain_id::text, connection_id, content, embedding::text FROM vdb.semantic_layer ORDER BY connection_id"
        ).fetchall()


def test_shared_projection_upsert_and_selected_delete(store: PostgresVDB) -> None:
    from auto_ontology.vdb.records import projection_id

    assert store.run(to_vdb_records([row("A"), row("B", "B", [0.0, 1.0, 0.0])])) == 2
    before = snapshot(store)
    assert [r[0] for r in before] == [
        projection_id("semantic_layer", cid, "shared-term") for cid in ("A", "B")
    ]
    assert store.projection_connection_ids("shared-term") == ["A", "B"]
    store.run(to_vdb_records([row("A", "changed", [0.0, 0.0, 1.0])]))
    after = snapshot(store)
    assert len(after) == 2 and after[1] == before[1]
    assert after[0][2:] != before[0][2:]
    assert [
        h["text"]
        for h in store.retrieval([[0.0, 1.0, 0.0]], where={"connection_id": "B"})[0]
    ] == ["B"]
    with pytest.raises(TypeError):
        store.delete_by_id("shared-term")
    with pytest.raises(ValueError, match="source_required"):
        store.delete_by_database("sales")
    assert snapshot(store) == after
    assert store.delete_by_id("shared-term", connection_id="A") == 1
    assert snapshot(store) == [before[1]]
    assert store.delete_by_connection("B") == [before[1][0]]


def test_missing_collection_read_delete_do_not_create(store: PostgresVDB) -> None:
    store.vector_size = 3
    assert store.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "A"}) == [[]]
    assert store.delete_by_connection("A") == []
    assert store.delete_by_id("shared-term", connection_id="A") == 0
    assert store.projection_connection_ids("shared-term") == []
    assert not store._table_exists()


def test_old_schema_requires_explicit_migration(store: PostgresVDB) -> None:
    with psycopg.connect(store.connection_string) as conn:
        conn.execute("CREATE SCHEMA vdb")
        conn.execute("CREATE TABLE vdb.semantic_layer (langchain_id uuid PRIMARY KEY)")
    with pytest.raises(ValueError, match="source_not_ready"):
        store.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "A"})
    with pytest.raises(ValueError, match="source_not_ready"):
        store.run(to_vdb_records([row("A")]))
    with pytest.raises(ValueError, match="source_not_ready"):
        store.delete_all()
    with pytest.raises(ValueError, match="source_not_ready"):
        store.delete_by_connection("A")


@pytest.mark.parametrize(
    "where", [None, {}, {"connection_id": None}, {"connection_id": ["A", "B"]}]
)
def test_unscoped_search_rejected_before_io(where: Any) -> None:
    store = PostgresVDB(connection_string="postgresql://127.0.0.1:1/unused")
    with pytest.raises(ValueError, match="source_required"):
        store.retrieval([[1.0, 0.0, 0.0]], where=where)


def test_scope_mismatch_and_reset_rejected_before_io() -> None:
    with pytest.raises(ValueError, match="source_required"):
        PostgresVDB(
            connection_string="postgresql://127.0.0.1:1/unused",
            database_name="sales",
            reset=True,
        )
    store = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", connection_id="A"
    )
    with pytest.raises(ValueError, match="source_conflict"):
        store.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "B"})
    with pytest.raises(ValueError, match="source_conflict"):
        store.run(to_vdb_records([row("B")]))


@pytest.mark.parametrize("key", ["id", "connection_id"])
def test_valid_embedded_rows_require_identity(key: str) -> None:
    value = row("A")
    del value["metadata"][key]
    with pytest.raises(ValueError):
        to_vdb_records([value])


def test_projection_identity_is_unambiguous() -> None:
    from auto_ontology.vdb.records import projection_id

    assert projection_id("a:b", "c", "d") != projection_id("a", "b:c", "d")
    [[record]] = to_vdb_records([row("A")])
    assert record["metadata"]["content_metadata"]["connection_id"] == "A"


@pytest.mark.parametrize("collection", ["data_objects_layer", "semantic_layer"])
def test_new_collections_have_source_column_index_and_keep_legacy_null_out_of_search(
    store: PostgresVDB, collection: str
) -> None:
    from psycopg import sql

    store.collection_name = collection
    store.run(to_vdb_records([row("A")]))
    with psycopg.connect(store.connection_string) as conn:
        assert conn.execute(
            "SELECT data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'vdb' AND table_name = %s AND column_name = 'connection_id'",
            (collection,),
        ).fetchone() == ("text", "YES")
        assert conn.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'vdb' AND tablename = %s AND indexdef LIKE '%%(connection_id)%%'",
            (collection,),
        ).fetchone()
        conn.execute(
            sql.SQL(
                "INSERT INTO vdb.{} (langchain_id, content, embedding) VALUES (gen_random_uuid(), 'unsafe legacy', '[1,0,0]')"
            ).format(sql.Identifier(collection))
        )
    assert [
        hit["text"]
        for hit in store.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "A"})[0]
    ] == ["A"]


def test_bound_reset_leaves_other_source_unchanged(store: PostgresVDB) -> None:
    store.run(to_vdb_records([row("A"), row("B", "B")]))
    before = snapshot(store)
    bound = PostgresVDB(
        connection_string=store.connection_string,
        schema_name="vdb",
        collection_name="semantic_layer",
        connection_id="A",
        reset=True,
    )
    try:
        assert snapshot(store) == [before[1]]
        assert bound.retrieval([[1.0, 0.0, 0.0]]) == [[]]
        with pytest.raises(ValueError, match="source_conflict"):
            bound.delete_by_id("shared-term", connection_id="B")
        assert bound.delete_all() == 0
        assert snapshot(store) == [before[1]]
    finally:
        bound.close()


@pytest.mark.parametrize("identity", [{"id": "node"}, {"connection_id": "A"}])
def test_direct_write_also_rejects_missing_identity(identity: dict) -> None:
    store = PostgresVDB(connection_string="postgresql://127.0.0.1:1/unused")
    with pytest.raises(ValueError, match="source_required"):
        store.write_to_index(
            [{"text": "x", "metadata": {**identity, "embedding": [1.0, 0.0, 0.0]}}]
        )


def test_factories_pass_source_scope_without_name_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auto_ontology.vdb as module

    calls = []
    monkeypatch.setattr(module, "get_postgres_connection_string", lambda: "test-store")
    monkeypatch.setattr(module, "PostgresVDB", lambda **kwargs: calls.append(kwargs))
    module.get_data_vdb(connection_id="A", database_name="sales", reset=True)
    module.get_semantic_vdb(connection_id="B", database_name="sales")
    assert [
        (c["connection_id"], c["database_name"], c["collection_name"], c["reset"])
        for c in calls
    ] == [
        ("A", "sales", "data_objects_layer", True),
        ("B", "sales", "semantic_layer", False),
    ]
