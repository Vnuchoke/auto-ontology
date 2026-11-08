# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, real backup/restore rehearsal on the explicitly owned local PG17."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

ROOT = Path(__file__).resolve().parents[2]
LEGACY = "99d4d343dd0b"
ADDITIVE = "c19a83e47b20"
HEAD = "e38c06a17f42"
DUMMY = "TEST_REHEARSAL_DUMMY"


def _connect(database: str, *, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(
        host="127.0.0.1",
        port=46019,
        user="identity_test",
        password="identity_test_only",
        dbname=database,
        autocommit=autocommit,
        connect_timeout=3,
    )


@pytest.fixture
def rehearsal_store(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, str, str, Path]]:
    if os.environ.get("CONNECTION_IDENTITY_REHEARSAL_TEST") != "1":
        pytest.skip("requires explicit disposable backup/restore authorization")
    for key, expected in {
        "CONNECTION_IDENTITY_MIGRATION_TEST": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "46019",
        "POSTGRES_DATABASE": "identity_test",
        "POSTGRES_USER": "identity_test",
        "POSTGRES_PASSWORD": "identity_test_only",
    }.items():
        assert os.environ.get(key) == expected, f"rehearsal_guard:{key}"
    container = os.environ.get("IDENTITY_REHEARSAL_CONTAINER", "")
    assert re.fullmatch(r"[0-9a-f]{64}", container), "rehearsal_guard:container_id"
    # Inspect only public identity/network metadata, never Config.Env.
    inspected = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "[{{json .Id}},{{json .Name}},{{json .Config.Image}},{{json .NetworkSettings.Ports}}]",
            container,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert inspected.returncode == 0, "rehearsal_guard:inspect_failed"
    assert json.loads(inspected.stdout) == [
        container,
        "/auto-ontology-connection-identity-tests",
        "pgvector/pgvector:pg17",
        {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "46019"}]},
    ], "rehearsal_guard:container_metadata"

    from auto_ontology.connectors.registry import invalidate_connectors_cache
    from auto_ontology.dal.session import dispose_engine

    invalidate_connectors_cache()
    dispose_engine()
    artifact_root = Path("/tmp/opencode/identity-migration-rehearsal")
    artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="run-", dir=artifact_root))
    databases = [f"identity_rehearsal_{uuid4().hex}" for _ in range(2)]
    created = []
    with _connect("identity_test", autocommit=True) as admin:
        try:
            for database in databases:
                admin.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
                created.append(database)
            monkeypatch.setenv("POSTGRES_DATABASE", databases[0])
            monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
            yield container, *databases, artifacts
        finally:
            invalidate_connectors_cache()
            dispose_engine()
            for database in reversed(created):
                admin.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(database))
                )
            (artifacts / "cleanup.txt").write_text("plain_drop_completed=2\n")


def _alembic(action: str, revision: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", action, revision],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _migration(*args: str) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, "-m", "dev_tools.migrate_connection_identity", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    safe_output = DUMMY not in result.stdout + result.stderr
    assert safe_output, "migration_leaked_dummy"
    assert result.stdout, "migration_missing_receipt"
    return result.returncode, json.loads(result.stdout)


def _fingerprint(database: str) -> str:
    """Hash every row and column shape in all three schemas without logging payloads."""
    with _connect(database) as conn:
        columns = conn.execute(
            "SELECT table_schema,table_name,column_name,data_type,is_nullable "
            "FROM information_schema.columns WHERE table_schema IN ('public','frontend','vdb') "
            "ORDER BY table_schema,table_name,ordinal_position"
        ).fetchall()
        tables = conn.execute(
            "SELECT schemaname,tablename FROM pg_tables "
            "WHERE schemaname IN ('public','frontend','vdb') ORDER BY 1,2"
        ).fetchall()
        rows = []
        for namespace, table in tables:
            values = conn.execute(
                sql.SQL(
                    "SELECT row_to_json(t)::text FROM {}.{} t ORDER BY row_to_json(t)::text"
                ).format(sql.Identifier(namespace), sql.Identifier(table))
            ).fetchall()
            rows.append((namespace, table, values))
    return hashlib.sha256(json.dumps([columns, rows]).encode()).hexdigest()


def _seed_legacy(database: str) -> None:
    assert _alembic("upgrade", LEGACY).returncode == 0, "legacy_upgrade_failed"
    with _connect(database) as conn:
        conn.execute(
            "INSERT INTO catalog_database(id,name,description,dialect,connection) "
            "VALUES ('local','sales','curated root','postgres',%s::jsonb), "
            "('vault','archive','curated archive','postgres',NULL)",
            (
                json.dumps(
                    {
                        "type": "postgresql",
                        "host": "invalid",
                        "port": 5432,
                        "user": "synthetic",
                        "database": "sales",
                        "password": DUMMY,
                    }
                ),
            ),
        )
        conn.execute("""
            INSERT INTO catalog_schema(id,database_id,name,description)
                VALUES ('schema','local','public','curated schema');
            INSERT INTO catalog_table(id,schema_id,name,description)
                VALUES ('table','schema','orders','curated table');
            INSERT INTO catalog_column(id,table_id,name,description)
                VALUES ('column','table','id','curated column');
            INSERT INTO sql_query(id,sql_full_query)
                VALUES ('owned-query','SELECT * FROM orders'),('unknown-query','SELECT 42');
            INSERT INTO sql_query__table(sql_query_id,table_id) VALUES ('owned-query','table');
            INSERT INTO sql_attribute(id,name) VALUES ('attribute','revenue');
            INSERT INTO sql_attribute__sql(attribute_id,sql_query_id) VALUES ('attribute','owned-query');
            CREATE SCHEMA frontend;
            CREATE TABLE frontend.conversations(id uuid PRIMARY KEY,user_id text,title text);
            CREATE TABLE frontend.messages(id uuid PRIMARY KEY,conversation_id uuid
                REFERENCES frontend.conversations(id),role text,content text);
            INSERT INTO frontend.conversations VALUES
                ('11111111-1111-4111-8111-111111111111','owner','legacy history');
            INSERT INTO frontend.messages VALUES
                ('22222222-2222-4222-8222-222222222222',
                 '11111111-1111-4111-8111-111111111111','user','preserved question');
            CREATE EXTENSION vector;
            CREATE SCHEMA vdb;
            CREATE TABLE vdb.data_objects_layer(langchain_id uuid PRIMARY KEY,
                content text,embedding vector(3),langchain_metadata jsonb,database_name text,label text);
            """)
        conn.execute(
            "INSERT INTO vdb.data_objects_layer VALUES (%s,'curated vector','[1,0,0]',%s::jsonb,'sales','Table')",
            (
                str(uuid4()),
                json.dumps(
                    {
                        "id": "table",
                        "label": "Table",
                        "database_name": "sales",
                        "content_metadata": {"id": "table"},
                    }
                ),
            ),
        )


def test_legacy_backup_backfill_duplicate_refusal_and_restore(
    rehearsal_store: tuple[str, str, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container, source, restored, artifacts = rehearsal_store
    _seed_legacy(source)
    before = _fingerprint(source)
    with _connect(source) as conn:
        inline_before = hashlib.sha256(
            conn.execute(
                "SELECT connection::text FROM catalog_database WHERE id='local'"
            )
            .fetchone()[0]
            .encode()
        ).hexdigest()
    backup = subprocess.run(
        [
            "docker",
            "exec",
            "-e",
            "PGPASSWORD=identity_test_only",
            container,
            "pg_dump",
            "-U",
            "identity_test",
            "-Fc",
            "-d",
            source,
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert backup.returncode == 0, "backup_failed"
    assert backup.stdout.startswith(b"PGDMP"), "backup_not_custom_archive"
    archive = artifacts / "pre-upgrade.dump"
    with archive.open("xb") as handle:
        archive.chmod(0o600)
        handle.write(backup.stdout)
    assert _fingerprint(source) == before, "backup_changed_source"

    assert _alembic("upgrade", ADDITIVE).returncode == 0, "additive_upgrade_failed"
    with _connect(source) as conn:
        # Mirror the nullable Prisma field, not a Prisma deployment acceptance test.
        conn.execute("ALTER TABLE frontend.conversations ADD COLUMN connection_id text")
    additive = _fingerprint(source)
    status, report = _migration("--check")
    assert status == 2
    assert {issue["code"] for issue in report["issues"]} == {"origin_required"}
    mapping = artifacts / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "vault": {
                    "configuration_origin": "managed",
                    "credential_ref": "legacy/exact-reference",
                }
            }
        )
    )
    mapping.chmod(0o600)
    status, report = _migration("--check", "--mapping", str(mapping))
    assert status == 0 and report["pending_updates"] > 0
    assert report["unresolved_sql"]["sql_query"] == ["unknown-query"]
    assert _fingerprint(source) == additive, "check_wrote_data"
    status, report = _migration("--apply", "--mapping", str(mapping))
    assert status == 0 and report["applied"] is True
    assert report["ready_for_querying"] is False
    applied = _fingerprint(source)
    status, report = _migration("--apply", "--mapping", str(mapping))
    assert status == 0 and report["pending_updates"] == 0
    assert _fingerprint(source) == applied, "repeat_apply_changed_data"

    from auto_ontology.vdb.records import projection_id

    with _connect(source) as conn:
        for table, expected in (
            (
                "catalog_database",
                [("local", "curated root"), ("vault", "curated archive")],
            ),
            ("catalog_schema", [("schema", "curated schema")]),
            ("catalog_table", [("table", "curated table")]),
            ("catalog_column", [("column", "curated column")]),
        ):
            assert (
                conn.execute(
                    sql.SQL("SELECT id,description FROM {} ORDER BY id").format(
                        sql.Identifier(table)
                    )
                ).fetchall()
                == expected
            )
        assert (
            hashlib.sha256(
                conn.execute(
                    "SELECT connection::text FROM catalog_database WHERE id='local'"
                )
                .fetchone()[0]
                .encode()
            ).hexdigest()
            == inline_before
        ), "inline_credentials_changed"
        assert conn.execute(
            "SELECT credential_ref FROM catalog_database WHERE id='vault'"
        ).fetchone() == ("legacy/exact-reference",)
        assert conn.execute(
            "SELECT connection_id FROM sql_query WHERE id='unknown-query'"
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT connection_id FROM sql_query WHERE id='owned-query'"
        ).fetchone() == ("local",)
        assert conn.execute(
            "SELECT connection_id FROM sql_attribute WHERE id='attribute'"
        ).fetchone() == ("local",)
        assert conn.execute(
            "SELECT connection_id FROM frontend.conversations"
        ).fetchone() == (None,)
        assert conn.execute("SELECT count(*) FROM frontend.messages").fetchone() == (1,)
        assert conn.execute(
            "SELECT langchain_id::text,connection_id,content,embedding::text FROM vdb.data_objects_layer"
        ).fetchone() == (
            projection_id("data_objects_layer", "local", "table"),
            "local",
            "curated vector",
            "[1,0,0]",
        )
        assert conn.execute("SELECT to_regclass('vdb.semantic_layer')").fetchone() == (
            None,
        )

    from auto_ontology.connectors import registry
    from auto_ontology.connectors.sources import (
        SourceSelectionError,
        list_active_sources,
    )
    from auto_ontology.retrieval.source_context import resolve_query_source

    connector = Mock(side_effect=AssertionError("source_connector_must_not_open"))
    monkeypatch.setattr(registry, "get_connector", connector)
    monkeypatch.setattr(registry, "create_connector", connector)
    assert {s.connection_id for s in list_active_sources()} == {"local", "vault"}
    with pytest.raises(SourceSelectionError, match="^source_not_ready$"):
        resolve_query_source(connection_id="local")
    connector.assert_not_called()
    assert _fingerprint(source) == applied, "readiness_wrote_data"

    assert _alembic("upgrade", HEAD).returncode == 0, "head_upgrade_failed"
    with _connect(source) as conn:
        conn.execute(
            "INSERT INTO catalog_database(id,name,display_label) VALUES ('second','sales','Sales other')"
        )
    duplicates = _fingerprint(source)
    downgrade = _alembic("downgrade", ADDITIVE)
    assert downgrade.returncode != 0, "unsafe_downgrade_accepted"
    assert re.search(r"duplicate|backup", downgrade.stderr, re.IGNORECASE) is not None
    assert _fingerprint(source) == duplicates, "failed_downgrade_changed_data"
    with _connect(source) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD,
        )
        assert conn.execute(
            "SELECT count(*) FROM catalog_database WHERE name='sales'"
        ).fetchone() == (2,)

    restore = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "-e",
            "PGPASSWORD=identity_test_only",
            container,
            "pg_restore",
            "--exit-on-error",
            "--no-owner",
            "--no-acl",
            "-U",
            "identity_test",
            "-d",
            restored,
        ],
        input=archive.read_bytes(),
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert restore.returncode == 0, "restore_failed"
    assert _fingerprint(restored) == before, "restored_snapshot_mismatch"
    assert _fingerprint(source) == duplicates, "restore_changed_original"
    with _connect(restored) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == (
            LEGACY,
        )
        assert conn.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_schema='frontend' AND table_name='conversations' AND column_name='connection_id'"
        ).fetchone() == (0,)
    (artifacts / "receipt.txt").write_text(
        f"backup_sha256={hashlib.sha256(backup.stdout).hexdigest()}\n"
        f"legacy_snapshot_sha256={before}\nrestored_snapshot_sha256={_fingerprint(restored)}\n"
        "backup_exit=0\nrestore_exit=0\noriginal_duplicates_preserved=2\n"
    )
