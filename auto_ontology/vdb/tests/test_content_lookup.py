# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PostgresVDB.ids_with_content_containing`` against a real collection table."""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb.postgres import PostgresVDB
from auto_ontology.vdb.records import projection_id

SOURCE = "source-a"


@pytest.fixture
def vdb():
    """A collection shaped like the one ``init_vectorstore_table`` creates."""
    name = f"lookup_{uuid.uuid4().hex[:8]}"
    try:
        conn = psycopg.connect(get_postgres_connection_string(), connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 -- no database, nothing to assert
        pytest.skip(f"postgres not reachable: {exc}")

    rows = [
        ("a-email", "ColumnAttribute: Email. Sample values: x@y.z."),
        ("a-status", "ColumnAttribute: Status. Sample values: open."),
        ("a-clean", "ColumnAttribute: Phone."),
        ("a-percent", "ColumnAttribute: Rate. 100% Sample values"),
    ]
    projections = [
        (
            projection_id(name, SOURCE, node_id),
            SOURCE,
            text,
            json.dumps({"id": node_id}),
        )
        for node_id, text in rows
    ]
    # Same node and stale text, but outside this source's exact projection.
    projections.append(
        (
            projection_id(name, "source-b", "a-clean"),
            "source-b",
            "Sample values: stale",
            json.dumps({"id": "a-clean"}),
        )
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS vdb")
            cur.execute(
                f"CREATE TABLE vdb.{name} ("
                "  langchain_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  connection_id text,"
                "  content text,"
                "  langchain_metadata jsonb)"
            )
            cur.executemany(
                f"INSERT INTO vdb.{name} (langchain_id, connection_id, content, langchain_metadata) "
                "VALUES (%s, %s, %s, %s)",
                projections,
            )
        conn.commit()
    try:
        yield PostgresVDB(
            connection_string=get_postgres_connection_string(),
            collection_name=name,
            schema_name="vdb",
        )
    finally:
        with psycopg.connect(get_postgres_connection_string()) as cleanup:
            with cleanup.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS vdb.{name}")
            cleanup.commit()


def test_only_the_named_rows_holding_the_fragment(vdb: PostgresVDB) -> None:
    found = vdb.ids_with_content_containing(
        ["a-email", "a-clean", "a-missing"], "Sample values:", connection_id=SOURCE
    )

    assert found == ["a-email"]


def test_the_fragment_is_literal(vdb: PostgresVDB) -> None:
    """``%`` would match anything under ``LIKE``."""
    assert vdb.ids_with_content_containing(
        ["a-email", "a-percent"], "0%", connection_id=SOURCE
    ) == ["a-percent"]


def test_no_ids_asks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    vdb = PostgresVDB(connection_string="unused", connection_id=SOURCE)
    monkeypatch.setattr(vdb, "_table_exists", lambda: pytest.fail("unexpected I/O"))
    assert (
        vdb.ids_with_content_containing([], "Sample values:", connection_id=SOURCE)
        == []
    )
    with pytest.raises(ValueError, match="source_conflict"):
        vdb.ids_with_content_containing(
            ["a-email"], "Sample values:", connection_id="source-b"
        )


def test_a_collection_not_yet_created_holds_nothing() -> None:
    try:
        psycopg.connect(get_postgres_connection_string(), connect_timeout=3).close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"postgres not reachable: {exc}")
    vdb = PostgresVDB(
        connection_string=get_postgres_connection_string(),
        collection_name=f"absent_{uuid.uuid4().hex[:8]}",
        schema_name="vdb",
    )

    assert (
        vdb.ids_with_content_containing(
            ["a-email"], "Sample values:", connection_id=SOURCE
        )
        == []
    )
