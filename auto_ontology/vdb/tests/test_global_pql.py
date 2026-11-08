# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global PQL is an explicit, label-isolated exception to source projections."""

from collections.abc import Iterator

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import write_embedded_rows


@pytest.fixture(autouse=True)
def forbid_negative_test_io(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "clients" not in request.fixturenames:
        monkeypatch.setattr(
            psycopg,
            "connect",
            lambda *a, **k: pytest.fail("SQL called before validation"),
        )


def pql_row(label: str = "PqlAnalysis", **metadata: object) -> dict:
    return {
        "text": "question",
        "metadata": {
            "id": "shared",
            "label": label,
            "embedding": [1.0, 0.0, 0.0],
            **metadata,
        },
    }


@pytest.fixture
def clients(identity_database: None) -> Iterator[tuple[PostgresVDB, PostgresVDB]]:
    from alembic import command
    from alembic.config import Config
    from auto_ontology.dal.tests.test_connection_identity_schema import ROOT
    from auto_ontology.dal.session import dispose_engine

    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    kwargs = {
        "connection_string": get_postgres_connection_string(),
        "schema_name": "vdb",
        "collection_name": "semantic_layer",
    }
    source = PostgresVDB(**kwargs)
    pql = PostgresVDB(**kwargs, global_pql=True)
    try:
        yield source, pql
    finally:
        source.close()
        pql.close()
        dispose_engine()


def test_global_pql_upsert_search_and_delete_are_isolated(clients: tuple) -> None:
    from auto_ontology.vdb.records import global_pql_projection_id

    source, pql = clients
    write_embedded_rows(
        source, [pql_row("Term", connection_id="A"), pql_row("Term", connection_id="B")]
    )
    assert write_embedded_rows(pql, [pql_row()]) == 1
    changed = pql_row()
    changed["text"] = "changed question"
    write_embedded_rows(pql, [changed])
    with psycopg.connect(source.connection_string) as conn:
        rows = conn.execute(
            "SELECT langchain_id::text, connection_id, label, content FROM vdb.semantic_layer ORDER BY label, connection_id"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0] == (
            global_pql_projection_id("semantic_layer", "shared"),
            None,
            "PqlAnalysis",
            "changed question",
        )
        # Even a corrupt source-owned PQL row must not leak into global search.
        conn.execute(
            "INSERT INTO vdb.semantic_layer (langchain_id, connection_id, label, content, embedding) VALUES (gen_random_uuid(), 'C', 'PqlAnalysis', 'spoof', '[1,0,0]')"
        )
        conn.execute(
            "INSERT INTO vdb.semantic_layer (langchain_id, label, content, embedding) VALUES (gen_random_uuid(), 'Term', 'quarantined term', '[1,0,0]'), (gen_random_uuid(), 'SqlAttribute', 'quarantined SQL', '[1,0,0]')"
        )
    assert [hit["text"] for hit in pql.retrieval([[1.0, 0.0, 0.0]])[0]] == [
        "changed question"
    ]
    assert pql.delete_by_id("shared") == 1
    assert [
        hit["metadata"]["label"]
        for hit in source.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "B"})[0]
    ] == ["Term"]
    write_embedded_rows(pql, [pql_row()])
    source.delete_by_connection("A")
    assert len(pql.retrieval([[1.0, 0.0, 0.0]])[0]) == 1
    assert pql.delete_all() == 1
    assert (
        len(source.retrieval([[1.0, 0.0, 0.0]], where={"connection_id": "B"})[0]) == 1
    )
    with psycopg.connect(source.connection_string) as conn:
        assert conn.execute(
            "SELECT count(*) FROM vdb.semantic_layer WHERE connection_id IS NULL AND label <> 'PqlAnalysis'"
        ).fetchone() == (2,)


@pytest.mark.parametrize(
    "metadata",
    [
        {"label": "Term"},
        {"label": "SqlAttribute"},
        {"connection_id": "A"},
        {"content_metadata": {"connection_id": "A"}},
    ],
)
def test_global_mode_rejects_spoof_before_embedding(
    monkeypatch: pytest.MonkeyPatch, metadata: dict
) -> None:
    from auto_ontology.utils import embedding

    pql = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", global_pql=True
    )
    monkeypatch.setattr(
        embedding, "embed_rows", lambda *a, **k: pytest.fail("embedding called")
    )
    with pytest.raises(ValueError):
        embedding.embed_rows_into_vdb(
            [pql_row(**metadata)], None, pql, what="PqlAnalysis"
        )
    with pytest.raises(ValueError):
        pql.write_to_index([pql_row(**metadata)])
    with pytest.raises(ValueError):
        write_embedded_rows(pql, [pql_row(**metadata)])


@pytest.mark.parametrize(
    "where",
    [
        {"label": "Term"},
        {"label": {"$ne": "Term"}},
        {"connection_id": "A"},
        {"$or": [{"label": "Term"}]},
    ],
)
def test_global_filter_tampering_rejected_before_io(where: dict) -> None:
    pql = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", global_pql=True
    )
    with pytest.raises(ValueError):
        pql.retrieval([[1.0, 0.0, 0.0]], where=where)


def test_global_mode_cannot_reset_a_source_and_ordinary_mode_still_requires_source() -> (
    None
):
    pql = PostgresVDB(
        connection_string="postgresql://127.0.0.1:1/unused", global_pql=True
    )
    with pytest.raises(ValueError):
        pql.delete_by_connection("A")
    with pytest.raises(ValueError):
        pql.delete_by_id("shared", connection_id="A")
    with pytest.raises(ValueError):
        pql.projection_connection_ids("shared")
    with pytest.raises(ValueError):
        write_embedded_rows(
            PostgresVDB(connection_string="postgresql://127.0.0.1:1/unused"),
            [pql_row()],
        )
    with pytest.raises(ValueError):
        PostgresVDB(
            connection_string="postgresql://127.0.0.1:1/unused",
            global_pql=True,
            connection_id="A",
        )


def test_global_factory_is_explicit_and_keeps_semantic_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auto_ontology.vdb as factory

    monkeypatch.setattr(
        factory,
        "get_postgres_connection_string",
        lambda: "postgresql://127.0.0.1:1/unused",
    )
    client = factory.get_global_pql_vdb()
    assert client.global_pql and client.connection_id is None
    assert client.collection_name == factory.SEMANTIC_VDB_COLLECTION
    assert not factory.get_semantic_vdb().global_pql


def test_global_pql_crud_and_actual_example_retrieval(
    clients: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    import auto_ontology.vdb as factory
    import auto_ontology.utils as utils
    from auto_ontology.utils import embedding, retriever
    from auto_ontology.server.pql_analyses import service
    from auto_ontology.retrieval.kumo import rag

    source, pql = clients

    def embedded(rows: list[dict], *args: object, **kwargs: object) -> list[dict]:
        for row in rows:
            row["metadata"]["embedding"] = [1.0, 0.0, 0.0]
        return rows

    monkeypatch.setattr(factory, "get_global_pql_vdb", lambda: pql)
    monkeypatch.setattr(rag, "get_global_pql_vdb", lambda: pql)
    monkeypatch.setattr(utils, "get_embed_params", lambda: None)
    monkeypatch.setattr(embedding, "embed_rows", embedded)
    monkeypatch.setattr(retriever, "embed_texts", lambda *a, **kw: [[1.0, 0.0, 0.0]])
    created = service.create_pql_analysis(
        "Which customers?", "reason", "PREDICT target"
    )
    assert service.update_pql_analysis(
        created["id"], "Which updated customers?", "updated reason", "PREDICT updated"
    )
    query_client = retriever.Retriever(source, None)
    assert rag.fetch_pql_examples(query_client, "customers?") == [
        {
            "question": "Which updated customers?",
            "reasoning": "updated reason",
            "query": "PREDICT updated",
        }
    ]
    with psycopg.connect(source.connection_string) as conn:
        assert conn.execute(
            "SELECT count(*) FROM vdb.semantic_layer WHERE label = 'PqlAnalysis'"
        ).fetchone() == (1,)
    assert service.delete_pql_analysis(created["id"]) == {"id": created["id"]}
    assert rag.fetch_pql_examples(query_client, "customers?") == []
