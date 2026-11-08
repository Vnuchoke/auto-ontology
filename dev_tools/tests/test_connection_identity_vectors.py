# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from copy import deepcopy
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)
from auto_ontology.infra.postgres import get_postgres_connection_string
from dev_tools.tests.test_connection_identity_migration import _run


def _legacy_vector_store(metadata_type: str = "jsonb") -> dict[str, str]:
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    rows = {owner: str(uuid4()) for owner in ("a", "b")}
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE SCHEMA vdb")
        conn.execute(
            f"CREATE TABLE vdb.data_objects_layer (langchain_id uuid PRIMARY KEY, content text, embedding vector(3), langchain_metadata {metadata_type}, database_name text, label text)"
        )
        for owner, row_id in rows.items():
            conn.execute(
                "INSERT INTO catalog_database(id,name,display_label,configuration_origin,connection) VALUES (%s,'sales',%s,'managed',%s::jsonb)",
                (
                    owner,
                    owner,
                    json.dumps(
                        {
                            "type": "postgresql",
                            "database": "sales",
                            "password": "TEST_DUMMY_SECRET",
                        }
                    ),
                ),
            )
            conn.execute(
                "INSERT INTO catalog_schema(id,name,database_id) VALUES (%s,'public',%s)",
                (f"s-{owner}", owner),
            )
            conn.execute(
                "INSERT INTO catalog_table(id,name,schema_id) VALUES (%s,'orders',%s)",
                (f"t-{owner}", f"s-{owner}"),
            )
            conn.execute(
                "INSERT INTO vdb.data_objects_layer VALUES (%s,%s,'[1,0,0]',%s::jsonb,'sales','Table')",
                (
                    row_id,
                    f"{owner} data text",
                    json.dumps(
                        {
                            "id": f"t-{owner}",
                            "label": "Table",
                            "database_name": "sales",
                            "content_metadata": {"id": f"t-{owner}"},
                        }
                    ),
                ),
            )
    return rows


@pytest.mark.usefixtures("identity_database")
@pytest.mark.parametrize("metadata_type", ["json", "jsonb"])
def test_vector_check_is_read_only_and_data_backfill_keeps_embeddings(
    metadata_type: str,
) -> None:
    _legacy_vector_store(metadata_type)
    status, report = _run("--check")
    assert status == 0
    assert report["vectors"]["data_objects_layer"]["pending_updates"] == 2
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema='vdb' AND table_name='data_objects_layer' AND column_name='connection_id'"
            ).fetchone()[0]
            == 0
        )
    status, applied = _run("--apply")
    assert status == 0 and applied["vector_migration_complete"] is True
    from auto_ontology.vdb.records import projection_id

    with psycopg.connect(get_postgres_connection_string()) as conn:
        rows = conn.execute(
            "SELECT langchain_id::text,connection_id,embedding::text,langchain_metadata FROM vdb.data_objects_layer ORDER BY connection_id"
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows] == [
            (projection_id("data_objects_layer", owner, f"t-{owner}"), owner, "[1,0,0]")
            for owner in ("a", "b")
        ]
        assert all(
            row[3]["content_metadata"]["connection_id"] == row[1] for row in rows
        )
        before = rows
    status, report = _run("--apply")
    assert (
        status == 0 and report["vectors"]["data_objects_layer"]["pending_updates"] == 0
    )
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert (
            conn.execute(
                "SELECT langchain_id::text,connection_id,embedding::text,langchain_metadata FROM vdb.data_objects_layer ORDER BY connection_id"
            ).fetchall()
            == before
        )


@pytest.mark.usefixtures("identity_database")
def test_missing_collections_are_not_created_with_guessed_dimensions() -> None:
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    status, report = _run("--apply")
    assert status == 0
    assert report["ready_for_querying"] is False
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='vdb'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.usefixtures("identity_database")
def test_unknown_data_vectors_do_not_get_owner_from_ambiguous_name() -> None:
    _legacy_vector_store()
    unknown_id = str(uuid4())
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(
            "INSERT INTO vdb.data_objects_layer VALUES (%s,'unknown source','[0,1,0]',%s::jsonb,'sales','Table')",
            (
                unknown_id,
                json.dumps(
                    {"id": "unproven", "database_name": "sales", "label": "Table"}
                ),
            ),
        )
    status, report = _run("--apply")
    assert status == 0
    assert report["vectors"]["data_objects_layer"]["unresolved_ids"] == [unknown_id]
    with psycopg.connect(get_postgres_connection_string()) as conn:
        row = conn.execute(
            "SELECT connection_id,embedding::text FROM vdb.data_objects_layer WHERE langchain_id=%s",
            (unknown_id,),
        ).fetchone()
        assert row == (None, "[0,1,0]")


@pytest.fixture
def semantic_store(identity_database: None) -> Iterator[str]:
    from auto_ontology.dal.session import dispose_engine

    dispose_engine()
    _legacy_vector_store()
    legacy_id = str(uuid4())
    try:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            conn.execute(
                "CREATE TABLE vdb.semantic_layer (langchain_id uuid PRIMARY KEY, content text, embedding vector(3), langchain_metadata jsonb, database_name text, label text)"
            )
            conn.execute(
                "INSERT INTO vdb.semantic_layer VALUES (%s,'OLD MIXED: LOCAL_SAMPLE REMOTE_SAMPLE','[0,1,0]',%s::jsonb,'sales','Term')",
                (legacy_id, json.dumps({"id": "shared", "label": "Term"})),
            )
            conn.execute(
                "INSERT INTO term(id,name,description,source) VALUES ('shared','Revenue','Global vocabulary','semantic')"
            )
            for owner, marker in (("a", "LOCAL_SAMPLE"), ("b", "REMOTE_SAMPLE")):
                conn.execute(
                    "INSERT INTO catalog_column(id,name,table_id,data_type,sample_values) VALUES (%s,'amount',%s,'text',%s)",
                    (f"c-{owner}", f"t-{owner}", json.dumps([marker])),
                )
                conn.execute(
                    "INSERT INTO table__term(table_id,term_id) VALUES (%s,'shared')",
                    (f"t-{owner}",),
                )
                conn.execute(
                    "INSERT INTO column_attribute(id,name,term_name,table_id,source_column,source,datatype) VALUES (%s,'amount','Revenue',%s,'amount','semantic','text')",
                    (f"attr-{owner}", f"t-{owner}"),
                )
                conn.execute(
                    "INSERT INTO column_attribute__term(attribute_id,term_id) VALUES (%s,'shared')",
                    (f"attr-{owner}",),
                )
                conn.execute(
                    "INSERT INTO column__has_attribute(column_id,attribute_id) VALUES (%s,%s)",
                    (f"c-{owner}", f"attr-{owner}"),
                )
        yield legacy_id
    finally:
        dispose_engine()


def test_mixed_semantic_text_requires_scoped_rebuild_and_repeat_is_model_free(
    semantic_store: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pandas as pd
    from auto_ontology.semantic import embed
    from dev_tools import migrate_connection_identity as migration

    captured = []

    def deterministic(rows: list[dict], params: object) -> pd.DataFrame:
        captured.extend(deepcopy(rows))
        return pd.DataFrame(
            [
                {**row, "metadata": {**row["metadata"], "embedding": [1.0, 0.0, 0.0]}}
                for row in rows
            ]
        )

    monkeypatch.setattr(
        embed,
        "resolve",
        lambda group, key: {
            "API_KEY": "TEST_DUMMY",
            "MODEL": "deterministic",
            "ENDPOINT": "http://127.0.0.1/unused",
        }[key],
    )
    monkeypatch.setattr(embed, "batch_embed", deterministic)
    assert migration.main(["--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["semantic_rebuild_required"] == ["a", "b"]
    assert not captured
    assert migration.main(["--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["vector_migration_complete"] is True, json.dumps(report)
    assert report["semantic_rebuild_required"] == []
    assert len(captured) == 4
    for row in captured:
        other = (
            "REMOTE_SAMPLE"
            if row["metadata"]["connection_id"] == "a"
            else "LOCAL_SAMPLE"
        )
        assert other not in row["text"]
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert conn.execute(
            "SELECT connection_id,content,embedding::text FROM vdb.semantic_layer WHERE langchain_id=%s",
            (semantic_store,),
        ).fetchone() == (None, "OLD MIXED: LOCAL_SAMPLE REMOTE_SAMPLE", "[0,1,0]")
        assert (
            conn.execute(
                "SELECT count(*) FROM vdb.semantic_layer WHERE connection_id IS NOT NULL"
            ).fetchone()[0]
            == 4
        )
    assert migration.main(["--apply"]) == 0
    capsys.readouterr()
    assert len(captured) == 4


def test_failed_semantic_rebuild_keeps_legacy_out_of_execution(
    semantic_store: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dev_tools import migrate_connection_identity as migration

    def unavailable(*args: object) -> bool:
        raise RuntimeError("PRIVATE_EMBEDDING_URL")

    monkeypatch.setattr(migration, "rebuild_semantic", unavailable)
    assert migration.main(["--apply"]) == 2
    output = capsys.readouterr()
    assert "PRIVATE_EMBEDDING_URL" not in output.out + output.err
    report = json.loads(output.out)
    assert report["applied"] is True
    assert report["ready_for_querying"] is False
    assert report["vector_migration_complete"] is False
    assert report["semantic_rebuild_failed"] == ["a", "b"]
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert conn.execute(
            "SELECT connection_id FROM vdb.semantic_layer WHERE langchain_id=%s",
            (semantic_store,),
        ).fetchone() == (None,)


def _pql_store(metadata_type: str = "jsonb", label_storage: str = "promoted") -> str:
    _legacy_vector_store(metadata_type)
    legacy_id = str(uuid4())
    metadata = {"id": "pql"}
    if label_storage == "json":
        metadata["label"] = "PqlAnalysis"
    elif label_storage == "nested":
        metadata["content_metadata"] = {"label": "PqlAnalysis"}
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute("INSERT INTO pql_analysis(id,name) VALUES ('pql','Global PQL')")
        conn.execute(
            f"CREATE TABLE vdb.semantic_layer (langchain_id uuid PRIMARY KEY, content text, embedding vector(3), langchain_metadata {metadata_type}, connection_id text)"
        )
        if label_storage == "promoted":
            conn.execute(
                "ALTER TABLE vdb.semantic_layer ADD COLUMN label text, ADD COLUMN database_name text"
            )
        conn.execute(
            "INSERT INTO vdb.semantic_layer(langchain_id,content,embedding,langchain_metadata) VALUES (%s,'PQL content','[0,1,0]',%s::jsonb)",
            (legacy_id, json.dumps(metadata)),
        )
        if label_storage == "promoted":
            conn.execute(
                "UPDATE vdb.semantic_layer SET label='PqlAnalysis',database_name='sales'"
            )
    return legacy_id


@pytest.mark.usefixtures("identity_database")
@pytest.mark.parametrize("metadata_type", ["json", "jsonb"])
@pytest.mark.parametrize("label_storage", ["promoted", "json", "nested"])
def test_global_pql_canonicalizes_promoted_or_json_label_without_changing_payload(
    metadata_type: str,
    label_storage: str,
) -> None:
    from auto_ontology.vdb.records import global_pql_projection_id

    _pql_store(metadata_type, label_storage)
    for _ in range(2):
        status, report = _run("--apply")
        assert status == 0
        assert report["vector_migration_complete"] is True
        with psycopg.connect(get_postgres_connection_string()) as conn:
            rows = conn.execute(
                "SELECT langchain_id::text,content,embedding::text,connection_id,langchain_metadata FROM vdb.semantic_layer"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][:4] == (
            global_pql_projection_id("semantic_layer", "pql"),
            "PQL content",
            "[0,1,0]",
            None,
        )
        if label_storage == "promoted":
            assert "label" not in rows[0][4]
            assert "database_name" not in rows[0][4]


@pytest.mark.usefixtures("identity_database")
@pytest.mark.parametrize("metadata_type", ["json", "jsonb"])
@pytest.mark.parametrize(
    "contradiction",
    [
        {"label": "Term"},
        {"content_metadata": {"label": "Term"}},
        {"connection_id": "a"},
        {"content_metadata": {"connection_id": "a"}},
        {"content_metadata": {"id": "other"}},
        {"database_name": "other"},
        {"content_metadata": {"database_name": "other"}},
    ],
)
def test_promoted_pql_rejects_contradictory_identity(
    metadata_type: str,
    contradiction: dict,
) -> None:
    legacy_id = _pql_store(metadata_type)
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(
            "UPDATE vdb.semantic_layer SET database_name='sales',langchain_metadata=%s::jsonb",
            (json.dumps({"id": "pql", **contradiction}),),
        )
    status, report = _run("--check")
    assert status == 2
    assert report["vectors"]["semantic_layer"]["conflict_ids"] == [legacy_id]


@pytest.mark.usefixtures("identity_database")
def test_global_pql_rejects_promoted_source_ownership() -> None:
    legacy_id = _pql_store()
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute("UPDATE vdb.semantic_layer SET connection_id='a'")
    status, report = _run("--check")
    assert status == 2
    assert report["vectors"]["semantic_layer"]["conflict_ids"] == [legacy_id]


@pytest.mark.usefixtures("identity_database")
@pytest.mark.parametrize("nested", [{"connection_id": "b"}, {"database_name": "other"}])
def test_data_rejects_nested_identity_conflicting_with_catalog(nested: dict) -> None:
    legacy_id = _legacy_vector_store()["a"]
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(
            "UPDATE vdb.data_objects_layer SET langchain_metadata=%s::jsonb WHERE langchain_id=%s",
            (json.dumps({"id": "t-a", "content_metadata": nested}), legacy_id),
        )
    status, report = _run("--check")
    assert status == 2
    assert report["vectors"]["data_objects_layer"]["conflict_ids"] == [legacy_id]


@pytest.mark.usefixtures("identity_database")
@pytest.mark.parametrize("collection", ["semantic_layer", "data_objects_layer"])
@pytest.mark.parametrize("same_node", [True, False])
def test_target_collisions_block_check_and_atomic_apply(
    collection: str,
    same_node: bool,
) -> None:
    from auto_ontology.vdb.records import global_pql_projection_id, projection_id

    if collection == "semantic_layer":
        legacy_id = _pql_store(label_storage="json")
        target = global_pql_projection_id(collection, "pql")
        node_id = "pql"
        label = "PqlAnalysis"
    else:
        legacy_id = _legacy_vector_store()["a"]
        target = projection_id(collection, "a", "t-a")
        node_id = "t-a"
        label = "Table"
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute("UPDATE catalog_database SET configuration_origin=NULL")
        conn.execute(
            f"INSERT INTO vdb.{collection}(langchain_id,content,embedding,langchain_metadata) VALUES (%s,'occupied','[0,0,1]',%s::jsonb)",
            (
                target,
                json.dumps(
                    {"id": node_id if same_node else "unrelated", "label": label}
                ),
            ),
        )
        before_vectors = conn.execute(
            f"SELECT row_to_json(t)::text FROM vdb.{collection} t ORDER BY langchain_id"
        ).fetchall()
        before_roots = conn.execute(
            "SELECT row_to_json(t)::text FROM catalog_database t ORDER BY id"
        ).fetchall()
    for mode in ("--check", "--apply"):
        status, report = _run(mode)
        assert status == 2
        assert report["applied"] is False
        assert set(report["vectors"][collection]["conflict_ids"]) == {legacy_id, target}
        with psycopg.connect(get_postgres_connection_string()) as conn:
            assert (
                conn.execute(
                    f"SELECT row_to_json(t)::text FROM vdb.{collection} t ORDER BY langchain_id"
                ).fetchall()
                == before_vectors
            )
            assert (
                conn.execute(
                    "SELECT row_to_json(t)::text FROM catalog_database t ORDER BY id"
                ).fetchall()
                == before_roots
            )


def test_successful_rebuild_without_writes_is_failed_and_retryable(
    semantic_store: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dev_tools import migrate_connection_identity as migration

    calls = []

    def no_writes(owner: str, name: str) -> bool:
        calls.append((owner, name))
        return True

    monkeypatch.setattr(migration, "rebuild_semantic", no_writes)
    for _ in range(2):
        assert migration.main(["--apply"]) == 2
        report = json.loads(capsys.readouterr().out)
        assert report["applied"] is True
        assert report["ready_for_querying"] is False
        assert report["vector_migration_complete"] is False
        assert report["semantic_rebuild_failed"] == ["a", "b"]
        assert report["semantic_rebuild_required"] == ["a", "b"]
        with psycopg.connect(get_postgres_connection_string()) as conn:
            assert conn.execute(
                "SELECT langchain_id::text,connection_id,content,embedding::text FROM vdb.semantic_layer"
            ).fetchall() == [
                (
                    semantic_store,
                    None,
                    "OLD MIXED: LOCAL_SAMPLE REMOTE_SAMPLE",
                    "[0,1,0]",
                )
            ]
    assert calls == [("a", "sales"), ("b", "sales")] * 2
