# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in lifecycle regression using owned loopback sources and fresh pgvector storage.

Only model computation and worker scheduling are adapted; source SQL, graphs,
catalog hydration, readiness, conversation persistence and HTTP/MCP stay real.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CONNECTION_IDENTITY_E2E_TEST") != "1",
    reason="requires explicit CONNECTION_IDENTITY_E2E_TEST=1 and owned loopback fixtures",
)

ROOT = Path(__file__).resolve().parents[3]
QUESTION = "What marker is stored in orders?"
QUERY = "SELECT marker FROM public.orders ORDER BY order_id"
QUERY_CALLS = [
    "QuestionIntentModel",
    "QuestionExtractionModel",
    "SQLGenerationModel",
    "IntentValidationModel",
]


class LocalModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.errors: list[str] = []

    def bind(self, **kwargs: Any) -> LocalModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        model = self

        class Structured:
            def invoke(self, messages: Any, **options: Any) -> Any:
                name = schema.__name__
                prompt = "\n".join(str(message.content) for message in messages)
                model.calls.append((name, prompt))
                lite = {
                    "sanitized_question": QUESTION,
                    "required_entity_name": ["marker"],
                }
                payloads = {
                    "QuestionExtractionLiteModel": lite,
                    "QuestionExtractionModel": {
                        **lite,
                        "subject": "orders",
                        "used_glossary_names": [],
                    },
                    "QuestionIntentModel": {
                        "question_type": "calculation",
                        "calculation_subtype": "match_based",
                        "rewritten_question": QUESTION,
                        "extracted_evidence": "",
                        "target_db": None,
                        "relevant_sql_example_numbers": [],
                    },
                    "SQLGenerationModel": {
                        "thought": "Read the stored marker; no assumptions.",
                        "sql_code": QUERY,
                        "response": "The stored marker is returned by this query.",
                    },
                    "IntentValidationModel": {
                        "is_valid": True,
                        "reasoning": "Direct projection.",
                        "join_issues": [],
                        "aggregation_issues": [],
                        "projection_issues": [],
                        "evidence_issues": [],
                    },
                    "FollowUpResolutionModel": {
                        "is_follow_up": True,
                        "standalone_question": QUESTION,
                    },
                }
                if name not in payloads or not prompt.strip():
                    model.errors.append(f"unexpected model call: {name}: {prompt}")
                    raise AssertionError(model.errors[-1])
                try:
                    return schema.model_validate(payloads[name])
                except Exception as exc:
                    model.errors.append(f"invalid model payload {name}: {exc}")
                    raise

        return Structured()

    def assert_calls(self, start: int, expected: list[str]) -> None:
        assert not self.errors, self.errors
        assert [name for name, _ in self.calls[start:]] == expected


def _embedding(
    texts: list[str], params: Any, input_type: str, **kwargs: Any
) -> list[list[float]]:
    # Known limitation: coverage's `score or inf` rejects exact zero; not fixed here.
    return [[1.0, 0.0 if input_type == "query" else 0.2, 0.0] for _ in texts]


@pytest.fixture
def lifecycle(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    # Every guard precedes metadata/source/socket I/O, including dependency imports.
    assert os.environ.get("CONNECTION_IDENTITY_E2E_TEST") == "1"
    assert os.environ.get("CONNECTION_IDENTITY_MIGRATION_TEST") == "1"
    assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    assert os.environ.get("POSTGRES_HOST") == "127.0.0.1"
    assert os.environ.get("POSTGRES_PORT") == "46019"
    assert os.environ.get("POSTGRES_DATABASE") == "identity_test"
    assert os.environ.get("POSTGRES_USER") == "identity_test"
    receipt = json.loads(
        Path(
            os.environ.get(
                "E2E_RESOURCES_RECEIPT", "/tmp/opencode/identity-e2e-resources.json"
            )
        ).read_text()
    )
    owner = receipt["owner"]
    assert isinstance(owner, str) and re.fullmatch(r"[0-9a-f]{8}", owner)
    assert len(receipt["sources"]) == 2
    urls: dict[str, str] = {}
    for entry in receipt["sources"]:
        key = entry["key"]
        assert key in {"a", "b"} and key not in urls
        url = os.environ[f"E2E_SOURCE_{key.upper()}_URL"]
        parsed = urlsplit(url)
        assert parsed.scheme == "postgresql"
        assert not parsed.query and not parsed.fragment
        assert (parsed.hostname, parsed.port, parsed.path, parsed.username) == (
            entry["host"],
            entry["port"],
            "/" + entry["database"],
            entry["user"],
        )
        assert (entry["host"], entry["database"], entry["user"]) == (
            "127.0.0.1",
            "sales",
            "source_test",
        )
        assert entry["name"] == f"auto-ontology-source-{key}-tests-{owner}"
        urls[key] = url
    assert set(urls) == {"a", "b"}

    import psycopg
    from alembic import command
    from alembic.config import Config
    from psycopg import sql

    from auto_ontology.connectors import registry, sources
    from auto_ontology.dal.session import dispose_engine
    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.utils import embedding, llm_invoke, retriever

    retriever.close_retrievers()
    registry.invalidate_connectors_cache()
    dispose_engine()
    database = f"identity_e2e_{uuid4().hex}"
    assert database.startswith("identity_e2e_")
    vdbs: list[Any] = []
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            monkeypatch.setattr(logger, "disabled", logger.disabled)
    model = LocalModel()
    monkeypatch.setattr(embedding, "embed_texts", _embedding)
    monkeypatch.setattr(retriever, "embed_texts", _embedding)
    monkeypatch.setattr(llm_invoke, "get_llm_client", lambda **kwargs: model)
    monkeypatch.setattr(
        llm_invoke, "get_non_reasoning_llm_client", lambda **kwargs: model
    )
    with psycopg.connect(get_postgres_connection_string(), autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        try:
            monkeypatch.setenv("POSTGRES_DATABASE", database)
            monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
            command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
            from auto_ontology.retrieval.entity_coverage import main as coverage
            from auto_ontology.retrieval.text_to_sql import main as query

            monkeypatch.setattr(coverage, "llm_client", model)
            monkeypatch.setattr(query, "llm_client", model)
            monkeypatch.setattr(query, "reasoning_llm_client", model)
            ids = {key: str(uuid4()) for key in urls}
            config = {
                key: url
                + "?"
                + urlencode(
                    {
                        "metadata_connection_id": ids[key],
                        "metadata_label": "East" if key == "a" else "West",
                    }
                )
                for key, url in urls.items()
            }
            monkeypatch.setenv("CONNECTION_STRINGS", ",".join(config.values()))
            discovered = sources.bootstrap_sources()
            assert {
                (s.connection_id, s.display_label, s.database_name) for s in discovered
            } == {(ids["a"], "East", "sales"), (ids["b"], "West", "sales")}
            with psycopg.connect(get_postgres_connection_string()) as conn:
                conn.execute("""
                    CREATE SCHEMA frontend;
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
                    CREATE TABLE frontend.acronyms (id uuid PRIMARY KEY, name text, description text,
                        created_at timestamptz NOT NULL DEFAULT NOW());
                    CREATE TABLE frontend.prompts (id uuid PRIMARY KEY, content text);
                    CREATE TABLE frontend.configurations (
                        key text PRIMARY KEY, value text NOT NULL,
                        updated_at timestamptz NOT NULL DEFAULT NOW());
                    """)
            world = {
                "ids": ids,
                "config": config,
                "urls": urls,
                "model": model,
                "vdbs": vdbs,
                "attributes": {},
                "tables": {},
                "columns": {},
            }
            _ingest(world, "a")
            _ingest(world, "b")
            from auto_ontology.dal import schema as s
            from auto_ontology.dal.session import store

            qid = str(uuid4())
            store().query_write(
                s.sql_query.insert().values(
                    id=qid, connection_id=ids["b"], sql_full_query=QUERY
                )
            )
            store().query_write(
                s.sql_query__table.insert().values(
                    sql_query_id=qid, table_id=world["tables"]["b"]
                )
            )
            yield world
            assert not model.errors, model.errors
        finally:
            retriever.close_retrievers()
            for vdb in vdbs:
                vdb.close()
            registry.invalidate_connectors_cache()
            dispose_engine()
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


def _ingest(world: dict[str, Any], key: str) -> None:
    from sqlalchemy import select

    from auto_ontology.catalog.ingest import ingest_catalog
    from auto_ontology.connectors.registry import get_connector
    from auto_ontology.connectors.postgres import PostgresDatabase
    from auto_ontology.dal import schema as s
    from auto_ontology.dal.session import store
    from auto_ontology.semantic.embed import SemanticEmbedder, embed_all_semantic_nodes
    from auto_ontology.utils.embedding import EmbedParams, embed_rows_into_vdb
    from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
    from auto_ontology.vdb import get_data_vdb, get_semantic_vdb

    cid = world["ids"][key]
    connector = get_connector(cid)
    assert isinstance(connector, PostgresDatabase)
    assert connector.connection_id == cid and connector.database_name == "sales"
    tables, columns = ingest_catalog(connector)
    assert set(tables["table_name"]) == {"orders"}
    assert set(columns["column_name"]) == (
        {"order_id", "marker", "draft_status"} if key == "a" else {"order_id", "marker"}
    )
    table_id = str(tables.iloc[0]["id"])
    marker_id = str(columns.loc[columns["column_name"] == "marker"].iloc[0]["id"])
    world["tables"][key], world["columns"][key] = table_id, marker_id
    terms = store().query_read(select(s.term.c.id).where(s.term.c.name == "orders"))
    term_id = terms[0]["id"] if terms else str(uuid4())
    if not terms:
        store().query_write(
            s.term.insert().values(
                id=term_id,
                name="orders",
                description="Stored orders",
                source="semantic",
            )
        )
    store().query_write(
        s.table__term.insert().values(table_id=table_id, term_id=term_id)
    )
    attr_id = str(uuid4())
    store().query_write(
        s.column_attribute.insert().values(
            id=attr_id,
            name="marker",
            source_column="marker",
            term_name="orders",
            table_id=table_id,
            description="Stored marker",
            source="semantic",
        )
    )
    store().query_write(
        s.column_attribute__term.insert().values(attribute_id=attr_id, term_id=term_id)
    )
    store().query_write(
        s.column__has_attribute.insert().values(
            column_id=marker_id, attribute_id=attr_id
        )
    )
    world["attributes"][key] = attr_id
    params = EmbedParams(model_name="local-test", embed_invoke_url="http://127.0.0.1:1")
    data = get_data_vdb(connection_id=cid, database_name="sales")
    semantic = get_semantic_vdb(connection_id=cid, database_name="sales")
    world["vdbs"].extend([data, semantic])
    rows = CatalogEmbeddingRowsOp(database_name="sales", connection_id=cid).process(
        (tables, columns)
    )
    assert embed_rows_into_vdb(
        rows.to_dict("records"), params, data, what="test catalog"
    ) == len(rows)
    assert (
        embed_all_semantic_nodes(SemanticEmbedder("sales", cid, params, semantic)) == 2
    )


class FiniteInbox:
    def __init__(self, payload: tuple[Any, ...]) -> None:
        self.payload: tuple[Any, ...] | None = payload

    def get(self) -> tuple[str, Any]:
        if self.payload is None:
            raise EOFError
        payload, self.payload = self.payload, None
        return "ask", payload


class EventJournal:
    def __init__(self) -> None:
        self.items: list[tuple[str, Any]] = []

    def put(self, item: tuple[str, Any]) -> None:
        self.items.append(item)


def _ask(
    world: dict[str, Any],
    key: str,
    history: list[dict] | None = None,
    *,
    question: str = QUESTION,
) -> dict:
    from auto_ontology.server.chat.worker import _worker_loop

    start = len(world["model"].calls)
    journal = EventJournal()
    _worker_loop(
        FiniteInbox(
            (question, False, None, None, history or [], None, world["ids"][key])
        ),
        journal,
    )
    world["model"].assert_calls(
        start, (["FollowUpResolutionModel"] if history else []) + QUERY_CALLS
    )
    assert not [value for tag, value in journal.items if tag == "error"], journal.items
    results = [
        value["answer"]
        for tag, value in journal.items
        if tag == "event" and value["type"] == "result"
    ]
    assert len(results) == 1, journal.items
    answer = results[0]
    assert answer["source"]["connection_id"] == world["ids"][key]
    assert answer["sql_code"].rstrip(";") == QUERY
    assert f"SOURCE_{key.upper()}_CONTROL" in json.dumps(
        answer["sql_response_from_db"]
    ), "source marker mismatch"
    prompts = [
        prompt
        for name, prompt in world["model"].calls[start:]
        if name == "SQLGenerationModel"
    ]
    assert len(prompts) == 1
    assert "orders" in prompts[0] and "marker" in prompts[0]
    assert "AVAILABLE TABLES (schema context):" in prompts[0]
    assert "No tables available." not in prompts[0]
    if key == "b":
        assert "draft_status" not in prompts[0]
    return answer


def _raw_scope(world: dict[str, Any], key: str) -> None:
    from auto_ontology.utils.retriever import (
        get_data_objects_retriever,
        get_semantic_objects_retriever,
    )

    cid = world["ids"][key]
    for retriever in (get_data_objects_retriever(), get_semantic_objects_retriever()):
        hits = retriever.query(
            QUESTION, top_k=100, vdb_kwargs={"where": {"connection_id": cid}}
        )
        assert hits, "raw vector oracle requires nonempty hits"
        assert {hit["metadata"]["connection_id"] for hit in hits} == {cid}, (
            "foreign raw vector hit"
        )
        expected = (
            world["tables"][key]
            if retriever.vdb.collection_name == "data_objects_layer"
            else world["attributes"][key]
        )
        assert expected in {hit["metadata"]["id"] for hit in hits}
        if key == "b":
            assert all("draft_status" not in hit["text"] for hit in hits)


def _snapshot(cid: str) -> dict[str, Any]:
    import psycopg
    from psycopg import sql
    from sqlalchemy import select

    from auto_ontology.dal import schema as s
    from auto_ontology.dal.session import store
    from auto_ontology.dal.terms import fetch_all_terms_and_attributes
    from auto_ontology.infra.postgres import get_postgres_connection_string

    schemas = select(s.catalog_schema.c.id).where(s.catalog_schema.c.database_id == cid)
    tables = select(s.catalog_table.c.id).where(
        s.catalog_table.c.schema_id.in_(schemas)
    )
    columns = select(s.catalog_column.c.id).where(
        s.catalog_column.c.table_id.in_(tables)
    )
    attrs = select(s.column_attribute.c.id).where(
        s.column_attribute.c.table_id.in_(tables)
    )
    queries = select(s.sql_query.c.id).where(s.sql_query.c.connection_id == cid)
    selections = {
        "root": select(s.catalog_database).where(s.catalog_database.c.id == cid),
        "schemas": select(s.catalog_schema).where(s.catalog_schema.c.id.in_(schemas)),
        "tables": select(s.catalog_table).where(s.catalog_table.c.id.in_(tables)),
        "columns": select(s.catalog_column).where(s.catalog_column.c.id.in_(columns)),
        "attributes": select(s.column_attribute).where(
            s.column_attribute.c.id.in_(attrs)
        ),
        "table_terms": select(s.table__term).where(
            s.table__term.c.table_id.in_(tables)
        ),
        "attribute_terms": select(s.column_attribute__term).where(
            s.column_attribute__term.c.attribute_id.in_(attrs)
        ),
        "column_attributes": select(s.column__has_attribute).where(
            s.column__has_attribute.c.column_id.in_(columns)
        ),
        "queries": select(s.sql_query).where(s.sql_query.c.id.in_(queries)),
        "query_tables": select(s.sql_query__table).where(
            s.sql_query__table.c.sql_query_id.in_(queries)
        ),
    }
    result = {
        name: sorted(store().query_read(statement), key=str)
        for name, statement in selections.items()
    }
    result["semantic"] = fetch_all_terms_and_attributes(connection_id=cid)
    with psycopg.connect(get_postgres_connection_string()) as conn:
        for collection in ("data_objects_layer", "semantic_layer"):
            result[collection] = conn.execute(
                sql.SQL(
                    "SELECT langchain_id, content, embedding::text, langchain_metadata::text, "
                    "connection_id, label FROM {} WHERE connection_id = %s ORDER BY langchain_id"
                ).format(sql.Identifier("vdb", collection)),
                (cid,),
            ).fetchall()
    return result


def test_same_name_connection_lifecycle(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg
    from sqlalchemy import null, update

    from auto_ontology.connectors import sources
    from auto_ontology.connectors.registry import (
        get_connector,
        invalidate_connectors_cache,
    )
    from auto_ontology.dal import schema as s
    from auto_ontology.dal.connections import (
        clear_connection,
        finalize_connection_removal,
    )
    from auto_ontology.dal.reset import delete_all_data
    from auto_ontology.dal.session import store
    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.retrieval.entity_coverage.main import get_coverage_response
    from auto_ontology.server.chat import conversation_dal
    from auto_ontology.utils.retriever import (
        get_data_objects_retriever,
        get_semantic_objects_retriever,
    )

    world = lifecycle
    assert world["tables"]["a"] != world["tables"]["b"]
    assert world["columns"]["a"] != world["columns"]["b"]
    for key in ("a", "b"):
        _raw_scope(world, key)
        start = len(world["model"].calls)
        response = get_coverage_response(
            {
                "question": QUESTION,
                "connection_id": world["ids"][key],
                "connectors": [get_connector(world["ids"][key])],
                "data_retriever": get_data_objects_retriever(),
                "semantic_retriever": get_semantic_objects_retriever(),
            }
        )
        world["model"].assert_calls(start, ["QuestionExtractionLiteModel"])
        assert response["coverage"] == 1.0, response
        assert {candidate["id"] for candidate in response["candidates"]} == {
            world["attributes"][key]
        }
        _ask(world, key)

    b_before = _snapshot(world["ids"]["b"])
    conversation = uuid4()
    first = conversation_dal.prepare_conversation(
        conversation_id=conversation,
        user_id="owner",
        question=QUESTION,
        source="app",
        connection_id=world["ids"]["a"],
    )
    assert first.history == []
    answer = _ask(world, "a")
    conversation_dal.persist_assistant_result(
        conversation_id=conversation,
        user_id="owner",
        connection_id=first.source.connection_id,
        analytics_id=first.analytics_id,
        response=answer["response"],
        sql_code=answer["sql_code"],
    )
    a_before = _snapshot(world["ids"]["a"])
    model_calls = len(world["model"].calls)
    # Env sources are renamed through their real metadata bootstrap, not managed CRUD.
    world["config"]["a"] = (
        world["urls"]["a"]
        + "?"
        + urlencode(
            {
                "metadata_connection_id": world["ids"]["a"],
                "metadata_label": "East renamed",
            }
        )
    )
    monkeypatch.setenv("CONNECTION_STRINGS", ",".join(world["config"].values()))
    sources.bootstrap_sources()
    a_after = _snapshot(world["ids"]["a"])
    assert a_after.pop("root")[0]["display_label"] == "East renamed"
    a_before.pop("root")
    assert a_after == a_before
    assert len(world["model"].calls) == model_calls
    assert _snapshot(world["ids"]["b"]) == b_before
    _raw_scope(world, "a")
    follow = conversation_dal.prepare_conversation(
        conversation_id=conversation, user_id="owner", question="And now?", source="app"
    )
    assert follow.source.connection_id == world["ids"]["a"]
    assert follow.source.display_label == "East renamed"
    assert len(follow.history) == 1 and follow.history[0].sql_code == QUERY
    history = [
        {
            "question": turn.question,
            "response": turn.response,
            "sql_code": turn.sql_code,
        }
        for turn in follow.history
    ]
    answer = _ask(world, "a", history, question="And now?")
    conversation_dal.persist_assistant_result(
        conversation_id=conversation,
        user_id="owner",
        connection_id=follow.source.connection_id,
        analytics_id=follow.analytics_id,
        response=answer["response"],
        sql_code=answer["sql_code"],
    )
    assert answer["source"]["display_label"] == "East renamed"

    with _http_server(world, monkeypatch) as url:
        _mcp_call(world, url, {"connection_id": world["ids"]["a"]}, "a")
        _mcp_call(world, url, {"connection_label": "West"}, "b")
        _mcp_call(world, url, {}, None)
        root_before = _snapshot(world["ids"]["b"])["root"]
        both = os.environ["CONNECTION_STRINGS"]
        monkeypatch.setenv("CONNECTION_STRINGS", world["config"]["b"])
        sources.bootstrap_sources()
        _mcp_call(world, url, {}, "b")
        monkeypatch.setenv("CONNECTION_STRINGS", both)
        sources.bootstrap_sources()
        assert _snapshot(world["ids"]["b"])["root"] == root_before

    assert _snapshot(world["ids"]["b"]) == b_before
    reset = delete_all_data(world["ids"]["a"])
    assert reset.connection_id == world["ids"]["a"]
    assert reset.data_rows > 0 and reset.semantic_rows > 0
    reset_a = _snapshot(world["ids"]["a"])
    assert reset_a["root"][0]["id"] == world["ids"]["a"]
    assert reset_a["root"][0]["display_label"] == "East renamed"
    assert reset_a["tables"] == reset_a["columns"] == []
    assert reset_a["data_objects_layer"] == reset_a["semantic_layer"] == []
    assert _snapshot(world["ids"]["b"]) == b_before
    _ask(world, "b")
    _ingest(world, "a")
    _raw_scope(world, "a")
    _ask(world, "a")
    assert _snapshot(world["ids"]["b"]) == b_before
    # Compose retirement locally: no external ingestion RPC or source SQL writes.
    monkeypatch.setenv("CONNECTION_STRINGS", world["config"]["b"])
    assert {source.connection_id for source in sources.list_active_sources()} == {
        world["ids"]["b"]
    }
    invalidate_connectors_cache()
    clear_connection(connection_id=world["ids"]["a"])
    assert _snapshot(world["ids"]["a"])["root"][0]["configuration_origin"] == "env"
    with pytest.raises(ValueError, match="source must be detached before removal"):
        finalize_connection_removal(world["ids"]["a"])
    # Fixture precondition only: removing env configuration does not detach its root.
    detached = store().query_write(
        update(s.catalog_database)
        .where(
            (s.catalog_database.c.id == world["ids"]["a"])
            & (s.catalog_database.c.configuration_origin == "env")
        )
        .values(connection=null(), credential_ref=None, configuration_origin=None)
        .returning(s.catalog_database.c.id)
    )
    assert detached == [{"id": world["ids"]["a"]}]
    assert _snapshot(world["ids"]["b"]) == b_before
    delete_all_data(world["ids"]["a"])
    finalize_connection_removal(world["ids"]["a"])
    assert _snapshot(world["ids"]["b"]) == b_before
    _ask(world, "b")
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert conn.execute(
            "SELECT connection_id FROM retired_connection WHERE connection_id = %s",
            (world["ids"]["a"],),
        ).fetchone() == (world["ids"]["a"],)
        assert (
            conn.execute(
                "SELECT id FROM catalog_database WHERE id = %s", (world["ids"]["a"],)
            ).fetchall()
            == []
        )
        assert conn.execute(
            "SELECT connection_id FROM frontend.conversations WHERE id = %s",
            (conversation,),
        ).fetchone() == (world["ids"]["a"],)
        assert conn.execute(
            "SELECT count(*) FROM frontend.messages WHERE conversation_id = %s",
            (conversation,),
        ).fetchone() == (4,)
    from auto_ontology.connectors.sources import SourceSelectionError

    with pytest.raises(SourceSelectionError, match="source_unavailable"):
        conversation_dal.prepare_conversation(
            conversation_id=conversation,
            user_id="owner",
            question=QUESTION,
            source="app",
        )


class InlineWorker:
    def __init__(self) -> None:
        self.journal = EventJournal()

    def submit(
        self,
        question: str,
        prediction: bool | None = None,
        target_db: str | None = None,
        subject_token: str | None = None,
        conversation_history: list[dict] | None = None,
        evidence: str | None = None,
        *,
        connection_id: str | None = None,
    ) -> None:
        from auto_ontology.server.chat.worker import _worker_loop

        _worker_loop(
            FiniteInbox(
                (
                    question,
                    prediction,
                    target_db,
                    subject_token,
                    conversation_history or [],
                    evidence,
                    connection_id,
                )
            ),
            self.journal,
        )

    def events(self) -> Iterator[dict]:
        for tag, value in self.journal.items:
            if tag == "error":
                raise RuntimeError(value)
            if tag == "event":
                yield value


class InlinePool:
    def acquire(self) -> InlineWorker:
        return InlineWorker()

    def return_alive(self, worker: InlineWorker) -> None:
        pass

    def replace(self, worker: InlineWorker) -> None:
        raise AssertionError("unexpected worker cancellation")


@contextmanager
def _http_server(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    import httpx
    import uvicorn
    from fastapi import FastAPI

    from auto_ontology.server.chat import router as chat_router
    from auto_ontology.server.connections.router import router as connections
    from auto_ontology.server.metadata.router import router as metadata

    pool = InlinePool()
    monkeypatch.setattr(chat_router, "get_pool", lambda: pool)
    app = FastAPI()
    for router in (connections, chat_router.router, metadata):
        app.include_router(router, prefix="/api")
    started = Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets: Any = None) -> None:
            await super().startup(sockets=sockets)
            started.set()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            assert started.wait(15), "HTTP fixture failed to start"
            response = httpx.get(url + "/api/connections/sources", trust_env=False)
            assert response.status_code == 200, response.text
            assert {item["connection_id"] for item in response.json()["data"]} == set(
                world["ids"].values()
            )
            assert (
                "password" not in response.text and "source_test" not in response.text
            )
            yield url
        finally:
            server.should_exit = True
            thread.join(15)
            assert not thread.is_alive(), "HTTP fixture failed to stop"


def _mcp_call(
    world: dict[str, Any], url: str, selectors: dict[str, str], expected: str | None
) -> None:
    script = """
import asyncio, json, sys
import httpx
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from auto_ontology_mcp.chat import register
from auto_ontology_mcp.config import Settings, DEFAULT_SPEC_PATH

async def main():
    url, selectors = sys.argv[1], json.loads(sys.argv[2])
    settings = Settings(api_url=url, spec_path=DEFAULT_SPEC_PATH, host="127.0.0.1",
                        port=3003, timeout_s=30, chat_timeout_s=60)
    async with httpx.AsyncClient(base_url=url, trust_env=False) as http:
        server = FastMCP("identity-e2e")
        register(server, settings, http)
        async with Client(server) as client:
            try:
                result = await client.call_tool("ask_question", {"question": sys.argv[3], **selectors})
            except ToolError as error:
                print(json.dumps({"error": str(error)}))
            else:
                print(json.dumps(result.structured_content))
asyncio.run(main())
"""
    start = len(world["model"].calls)
    process = subprocess.run(
        [
            str(ROOT / "mcp/.venv/bin/python"),
            "-c",
            script,
            url,
            json.dumps(selectors),
            QUESTION,
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "mcp")},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    world["model"].assert_calls(start, QUERY_CALLS if expected else [])
    if expected is None:
        assert "source_required" in result["error"]
    else:
        assert result["source"]["connection_id"] == world["ids"][expected]
        assert result["sql"].rstrip(";") == QUERY
        assert result["rows"] == [{"marker": f"SOURCE_{expected.upper()}_CONTROL"}]


def test_oracles_detect_missing_filter_and_connector_swap(
    lifecycle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg
    from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
    from psycopg import sql

    from auto_ontology.connectors import registry
    from auto_ontology.connectors.sources import SourceSelectionError
    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.retrieval.text_to_sql.main import _build_state
    from auto_ontology.utils.retriever import (
        get_data_objects_retriever,
        get_semantic_objects_retriever,
    )

    world = lifecycle
    _raw_scope(world, "a")
    _raw_scope(world, "b")
    # Place real foreign vectors nearer than A before disabling only the SQL CID conjunct.
    with psycopg.connect(get_postgres_connection_string()) as conn:
        for collection in ("data_objects_layer", "semantic_layer"):
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET embedding = '[1,0.01,0]'::vector WHERE connection_id = %s"
                ).format(sql.Identifier("vdb", collection)),
                (world["ids"]["b"],),
            )
    original = AsyncPGVectorStore._handle_field_filter

    def missing_cid(self: Any, field: str, value: Any) -> tuple[str, dict]:
        return (
            ("TRUE", {}) if field == "connection_id" else original(self, field, value)
        )

    with monkeypatch.context() as mutant:
        mutant.setattr(AsyncPGVectorStore, "_handle_field_filter", missing_cid)
        # The same positive oracle must fail specifically on a nonempty foreign hit.
        with pytest.raises(AssertionError, match="foreign raw vector hit"):
            _raw_scope(world, "a")
    _raw_scope(world, "a")
    _ask(world, "a")
    b = registry.get_connector(world["ids"]["b"])
    start = len(world["model"].calls)
    with pytest.raises(SourceSelectionError, match="source_unavailable"):
        _build_state(
            {
                "question": QUESTION,
                "connection_id": world["ids"]["a"],
                "connectors": [b],
                "data_retriever": get_data_objects_retriever(),
                "semantic_retriever": get_semantic_objects_retriever(),
            }
        )
    world["model"].assert_calls(start, [])
    swapped = registry.create_connector(world["urls"]["b"])
    swapped.connection_id = world["ids"]["a"]
    try:
        with monkeypatch.context() as mutant:
            mutant.setitem(registry._connectors, world["ids"]["a"], swapped)
            with pytest.raises(AssertionError, match="source marker mismatch"):
                _ask(world, "a")
    finally:
        swapped.close()
    _ask(world, "a")
