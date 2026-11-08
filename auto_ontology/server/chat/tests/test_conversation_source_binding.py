# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PostgreSQL acceptance, rollback and row-lock tests; never source SQL."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from time import monotonic
from unittest.mock import Mock
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from auto_ontology.connectors.sources import SourceDescriptor, SourceSelectionError
from auto_ontology.dal.session import dispose_engine
from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.retrieval import source_context
from auto_ontology.server.chat import conversation_dal as dal

ROOT = Path(__file__).resolve().parents[4]
A, B = str(uuid4()), str(uuid4())


@pytest.fixture
def identity_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if os.environ.get("CONNECTION_IDENTITY_MIGRATION_TEST") != "1":
        pytest.skip("requires an explicitly authorized disposable migration database")
    assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    assert os.environ.get("POSTGRES_HOST") in {"127.0.0.1", "localhost"}
    from alembic import command
    from alembic.config import Config

    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            monkeypatch.setattr(logger, "disabled", logger.disabled)
    dispose_engine()
    database = f"identity_chat_{uuid4().hex}"
    with psycopg.connect(get_postgres_connection_string(), autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        monkeypatch.setenv("POSTGRES_DATABASE", database)
        monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
        try:
            command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
            with psycopg.connect(get_postgres_connection_string()) as conn:
                conn.execute("CREATE SCHEMA frontend")
                conn.execute("""
                    CREATE TABLE frontend.conversations (
                        id uuid PRIMARY KEY, user_id text NOT NULL,
                        title text NOT NULL DEFAULT '', connection_id text,
                        created_at timestamptz NOT NULL DEFAULT NOW(),
                        updated_at timestamptz NOT NULL DEFAULT NOW());
                    CREATE TABLE frontend.messages (
                        id uuid PRIMARY KEY, conversation_id uuid NOT NULL
                            REFERENCES frontend.conversations(id) ON DELETE CASCADE,
                        role text NOT NULL, content text NOT NULL DEFAULT '',
                        sql_code text, sql_response text,
                        created_at timestamptz NOT NULL DEFAULT NOW());
                    CREATE TABLE frontend.conversation_analytics (
                        id uuid PRIMARY KEY, user_id text NOT NULL,
                        source text NOT NULL DEFAULT 'app', question text NOT NULL DEFAULT '',
                        question_timestamp timestamptz NOT NULL DEFAULT NOW(),
                        response text, response_timestamp timestamptz, sql text);
                    """)
                for cid, label in ((A, "Sales A"), (B, "Sales B")):
                    conn.execute(
                        "INSERT INTO catalog_database "
                        "(id, name, display_label, configuration_origin, dialect, connection) "
                        "VALUES (%s, 'sales', %s, 'managed', 'postgres', '{}'::jsonb)",
                        (cid, label),
                    )

            def semantic_ready(*, connection_id: str) -> bool:
                assert connection_id in {A, B}
                return True

            def vectors_ready(connection_id: str) -> None:
                assert connection_id in {A, B}

            monkeypatch.setattr(
                source_context, "semantic_layer_calculated", semantic_ready
            )
            monkeypatch.setattr(
                source_context, "require_source_vectors_ready", vectors_ready
            )
            yield
        finally:
            dispose_engine()
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )


def _seed(cid: UUID, *, source: str | None = None, role: str | None = None) -> None:
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(
            "INSERT INTO frontend.conversations (id, user_id, connection_id) VALUES (%s, 'owner', %s)",
            (cid, source),
        )
        if role is not None:
            conn.execute(
                "INSERT INTO frontend.messages (id, conversation_id, role, content) VALUES (%s, %s, %s, 'legacy')",
                (uuid4(), cid, role),
            )


def _snapshot() -> list[list[dict]]:
    with psycopg.connect(
        get_postgres_connection_string(), row_factory=dict_row
    ) as conn:
        return [
            list(conn.execute(f"SELECT * FROM frontend.{table} ORDER BY id").fetchall())
            for table in ("conversations", "messages", "conversation_analytics")
        ]


def _append(cid: UUID, *, manual: bool = False, **selectors: str) -> object:
    if manual:
        return dal.append_message(
            conversation_id=cid,
            user_id="owner",
            role="user",
            content="question",
            **selectors,
        )
    return dal.prepare_conversation(
        conversation_id=cid,
        user_id="owner",
        question="question",
        source="app",
        **selectors,
    )


@pytest.mark.parametrize("manual", [False, True])
def test_bind_inherit_conflict_and_atomic_rejection(
    identity_database: None, manual: bool
) -> None:
    cid = uuid4()
    _seed(cid)
    accepted = _append(cid, manual=manual, connection_label="Sales A")
    if not manual:
        assert accepted.source.connection_id == A
        assert accepted.history == []
    else:
        assert set(accepted) == {
            "id",
            "conversation_id",
            "role",
            "content",
            "sql_code",
            "sql_response",
            "created_at",
        }
    _append(cid, manual=manual)
    before = _snapshot()
    assert before[0][0]["connection_id"] == A
    assert len(before[1]) == 2
    assert len(before[2]) == (0 if manual else 2)
    with pytest.raises(SourceSelectionError) as error:
        _append(cid, manual=manual, connection_id=B)
    assert (error.value.status_code, error.value.code) == (409, "source_conflict")
    assert _snapshot() == before


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("role", ["user", "assistant", "chart", "error"])
def test_legacy_any_message_is_read_only(
    identity_database: None, manual: bool, role: str
) -> None:
    cid = uuid4()
    _seed(cid, role=role)
    before = _snapshot()
    with pytest.raises(SourceSelectionError, match="legacy_conversation_read_only"):
        _append(cid, manual=manual, connection_id=A)
    assert _snapshot() == before


@pytest.mark.parametrize("manual", [False, True])
def test_owner_denial_precedes_source_callback(
    identity_database: None, monkeypatch: pytest.MonkeyPatch, manual: bool
) -> None:
    cid = uuid4()
    _seed(cid, source=A)
    spy = Mock(side_effect=AssertionError("must not resolve foreign source"))
    monkeypatch.setattr(dal, "resolve_source", spy)
    monkeypatch.setattr(dal, "resolve_query_source", spy)
    before = _snapshot()
    with pytest.raises(dal.ConversationAccessError):
        if manual:
            dal.append_message(
                conversation_id=cid, user_id="foreign", role="user", connection_id=B
            )
        else:
            dal.prepare_conversation(
                conversation_id=cid,
                user_id="foreign",
                question="secret",
                source="app",
                connection_id=B,
            )
    spy.assert_not_called()
    assert _snapshot() == before


def test_manual_missing_never_creates(identity_database: None) -> None:
    with pytest.raises(dal.ConversationAccessError):
        _append(uuid4(), manual=True, connection_id=A)
    assert _snapshot() == [[], [], []]


@pytest.mark.parametrize(
    "selector,code,status",
    [
        ({}, "source_required", 422),
        ({"connection_id": "missing"}, "source_unavailable", 404),
        ({"target_db": "sales"}, "ambiguous_source", 409),
        (
            {"connection_id": A, "connection_label": "Sales A"},
            "invalid_source_selector",
            422,
        ),
    ],
)
def test_rejected_new_chat_rolls_back_creation(
    identity_database: None, selector: dict, code: str, status: int
) -> None:
    with pytest.raises(SourceSelectionError) as error:
        _append(uuid4(), **selector)
    assert (error.value.code, error.value.status_code) == (code, status)
    assert _snapshot() == [[], [], []]


def test_query_readiness_blocks_chat_but_not_manual_or_late_errors(
    identity_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        source_context,
        "require_source_vectors_ready",
        Mock(side_effect=RuntimeError("offline")),
    )
    cid = uuid4()
    with pytest.raises(SourceSelectionError, match="source_not_ready"):
        _append(cid, connection_id=A)
    assert _snapshot() == [[], [], []]
    _seed(cid)
    _append(cid, manual=True, connection_id=A)
    dal.persist_result_message(
        conversation_id=cid,
        user_id="owner",
        connection_id=A,
        content="Agent failed: offline",
        sql_response=None,
    )
    assert len(_snapshot()[1]) == 2


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_unavailable_bound_source_never_replaced(
    identity_database: None, manual: bool, deleted: bool
) -> None:
    cid = uuid4()
    _seed(cid, source=A)
    with psycopg.connect(get_postgres_connection_string()) as conn:
        if deleted:
            conn.execute("DELETE FROM catalog_database WHERE id = %s", (A,))
        else:
            conn.execute(
                "UPDATE catalog_database SET configuration_origin = NULL WHERE id = %s",
                (A,),
            )
    before = _snapshot()
    with pytest.raises(SourceSelectionError, match="source_unavailable") as error:
        _append(cid, manual=manual)
    assert error.value.status_code == 404
    assert _snapshot() == before


def test_analytics_failure_rolls_back_new_binding_and_message(
    identity_database: None,
) -> None:
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute(
            "ALTER TABLE frontend.conversation_analytics ADD CHECK (source <> 'rejected-channel')"
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        dal.prepare_conversation(
            conversation_id=uuid4(),
            user_id="owner",
            question="question",
            source="rejected-channel",
            connection_id=A,
        )
    assert _snapshot() == [[], [], []]


def test_successful_results_preserve_history_sql_and_analytics_channel(
    identity_database: None,
) -> None:
    cid = uuid4()
    accepted = _append(cid, connection_id=A)
    dal.persist_assistant_result(
        conversation_id=cid,
        user_id="owner",
        connection_id=A,
        analytics_id=accepted.analytics_id,
        response="answer",
        sql_code="SELECT 1",
    )
    dal.persist_result_message(
        conversation_id=cid,
        user_id="owner",
        connection_id=A,
        content="```chart\n{}",
        sql_response="[]",
    )
    followup = _append(cid)
    assert followup.source.connection_id == A
    assert followup.history == [dal.ConversationTurn("question", "answer", "SELECT 1")]
    analytics = next(
        row for row in _snapshot()[2] if str(row["id"]) == accepted.analytics_id
    )
    assert (analytics["source"], analytics["response"], analytics["sql"]) == (
        "app",
        "answer",
        "SELECT 1",
    )
    assert analytics["response_timestamp"] is not None


@pytest.mark.parametrize("result", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_late_owner_denial_or_missing_never_calls_source(
    identity_database: None,
    monkeypatch: pytest.MonkeyPatch,
    result: bool,
    missing: bool,
) -> None:
    cid = uuid4()
    if not missing:
        _seed(cid, source=A)
    spy = Mock(side_effect=AssertionError("must check owner before metadata"))
    monkeypatch.setattr(dal, "resolve_source", spy)
    before = _snapshot()
    with pytest.raises(dal.ConversationAccessError):
        if result:
            dal.persist_result_message(
                conversation_id=cid,
                user_id="foreign",
                connection_id=A,
                content="chart",
                sql_response="[]",
            )
        else:
            dal.persist_assistant_result(
                conversation_id=cid,
                user_id="foreign",
                connection_id=A,
                analytics_id=str(uuid4()),
                response="answer",
                sql_code=None,
            )
    spy.assert_not_called()
    assert _snapshot() == before


def test_stateless_analytics_remain_unbound(identity_database: None) -> None:
    analytics_id = dal.create_stateless_analytics(
        user_id="owner", question="question", source="mcp"
    )
    dal.persist_analytics_result(
        analytics_id=analytics_id,
        user_id="owner",
        response="answer",
        sql_code="SELECT 1",
    )
    conversations, messages, analytics = _snapshot()
    assert conversations == messages == []
    assert len(analytics) == 1
    assert (analytics[0]["source"], analytics[0]["response"], analytics[0]["sql"]) == (
        "mcp",
        "answer",
        "SELECT 1",
    )


@pytest.mark.parametrize("kind", ["mismatch", "inactive", "deleted"])
@pytest.mark.parametrize("result", [False, True])
def test_late_writes_reject_stale_source(
    identity_database: None, kind: str, result: bool
) -> None:
    cid = uuid4()
    accepted = _append(cid, connection_id=A)
    if kind != "mismatch":
        with psycopg.connect(get_postgres_connection_string()) as conn:
            if kind == "inactive":
                conn.execute(
                    "UPDATE catalog_database SET configuration_origin = NULL WHERE id = %s",
                    (A,),
                )
            else:
                conn.execute("DELETE FROM catalog_database WHERE id = %s", (A,))
    before = _snapshot()
    with pytest.raises(SourceSelectionError):
        if result:
            dal.persist_result_message(
                conversation_id=cid,
                user_id="owner",
                connection_id=B if kind == "mismatch" else A,
                content="```chart\n{}",
                sql_response="[]",
            )
        else:
            dal.persist_assistant_result(
                conversation_id=cid,
                user_id="owner",
                connection_id=B if kind == "mismatch" else A,
                analytics_id=accepted.analytics_id,
                response="Agent failed: offline",
                sql_code=None,
            )
    assert _snapshot() == before


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("same_source", [False, True])
def test_independent_first_turns_serialize(
    identity_database: None,
    monkeypatch: pytest.MonkeyPatch,
    manual: bool,
    same_source: bool,
) -> None:
    cid = uuid4()
    if manual:
        _seed(cid)
    start = Barrier(2)
    entered, release = Event(), Event()

    def hold_first(
        resolver: Callable[..., SourceDescriptor],
    ) -> Callable[..., SourceDescriptor]:
        def resolve(**selectors: str | None) -> SourceDescriptor:
            if not entered.is_set():
                entered.set()
                assert release.wait(10), "observer did not release the row lock"
            return resolver(**selectors)

        return resolve

    monkeypatch.setattr(dal, "resolve_source", hold_first(dal.resolve_source))
    monkeypatch.setattr(
        dal, "resolve_query_source", hold_first(dal.resolve_query_source)
    )

    def submit(is_manual: bool, source: str) -> str:
        start.wait(timeout=10)
        try:
            _append(cid, manual=is_manual, connection_id=source)
            return source
        except SourceSelectionError as error:
            assert (error.status_code, error.code) == (409, "source_conflict")
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(submit, False, A),
            pool.submit(submit, manual, A if same_source else B),
        ]
        try:
            assert entered.wait(10), (
                "first writer never reached locked source resolution"
            )
            deadline = monotonic() + 10
            with psycopg.connect(
                get_postgres_connection_string(), autocommit=True
            ) as observer:
                while monotonic() < deadline:
                    blocked = observer.execute(
                        "SELECT pid, unnest(pg_blocking_pids(pid)) FROM pg_stat_activity "
                        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                    ).fetchone()
                    if blocked is not None:
                        assert blocked[0] != blocked[1]
                        break
                else:
                    pytest.fail(
                        "second independent transaction never waited on the first"
                    )
        finally:
            release.set()
        outcomes = [future.result(timeout=20) for future in futures]
    rows = _snapshot()
    assert len(rows[0]) == 1
    if same_source:
        assert outcomes == [A, A]
        assert len(rows[1]) == 2
        assert len(rows[2]) == (1 if manual else 2)
    else:
        assert outcomes.count("conflict") == 1
        assert rows[0][0]["connection_id"] == next(
            value for value in outcomes if value != "conflict"
        )
        assert len(rows[1]) == 1
        assert len(rows[2]) == (0 if manual and outcomes[0] == "conflict" else 1)
