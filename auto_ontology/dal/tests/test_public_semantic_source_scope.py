# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public semantic projections against an explicitly disposable metadata store."""

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.dal import schema as s, terms
from auto_ontology.dal import exploration
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database,  # noqa: F401
)
from auto_ontology.connectors import sources
from auto_ontology.server.terms import router as term_router
from auto_ontology.server.sql_attributes import router as sql_router
from auto_ontology.server.exploration import router as exploration_router


@pytest.fixture
def semantic_sources(identity_database: None) -> Iterator[None]:  # noqa: F811
    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    for table, values in [
        (
            s.term,
            {
                "id": "shared",
                "name": "Common",
                "description": "Global meaning",
                "synonyms": ["Vocabulary"],
                "name_certified": True,
                "description_certified": True,
            },
        ),
        (s.term, {"id": "only-b", "name": "B only"}),
    ]:
        store().query_write(insert(table).values(**values))
    for cid in ("a", "b"):
        for table, values in [
            (
                s.catalog_database,
                {
                    "id": cid,
                    "name": "sales",
                    "display_label": cid,
                    "configuration_origin": "managed",
                },
            ),
            (
                s.catalog_schema,
                {"id": f"schema-{cid}", "name": "public", "database_id": cid},
            ),
            (
                s.catalog_table,
                {"id": f"table-{cid}", "name": "orders", "schema_id": f"schema-{cid}"},
            ),
            (s.table__term, {"table_id": f"table-{cid}", "term_id": "shared"}),
            (
                s.catalog_column,
                {
                    "id": f"column-{cid}",
                    "name": "value",
                    "table_id": f"table-{cid}",
                    "sample_values": [f"marker-{cid}"],
                },
            ),
            (
                s.column_attribute,
                {
                    "id": f"attr-{cid}",
                    "name": f"marker-{cid}",
                    "description": f"marker-{cid}",
                    "table_id": f"table-{cid}",
                    "source_column": "value",
                    "term_name": "Common",
                    "source": "semantic",
                    "certified": cid == "a",
                },
            ),
            (
                s.column_attribute__term,
                {"attribute_id": f"attr-{cid}", "term_id": "shared"},
            ),
            (
                s.column__has_attribute,
                {"attribute_id": f"attr-{cid}", "column_id": f"column-{cid}"},
            ),
            (
                s.sql_query,
                {
                    "id": f"sql-{cid}",
                    "connection_id": cid,
                    "sql_full_query": f"SELECT '{cid}' FROM public.orders",
                },
            ),
            (
                s.sql_query__table,
                {"sql_query_id": f"sql-{cid}", "table_id": f"table-{cid}"},
            ),
            (
                s.sql_attribute,
                {
                    "id": f"sql-attr-{cid}",
                    "connection_id": cid,
                    "name": f"sql-marker-{cid}",
                    "description": f"sql-marker-{cid}",
                    "expression": "count(*)",
                    "certified": cid == "a",
                },
            ),
            (
                s.sql_attribute__term,
                {"attribute_id": f"sql-attr-{cid}", "term_id": "shared"},
            ),
            (
                s.sql_attribute__sql,
                {"attribute_id": f"sql-attr-{cid}", "sql_query_id": f"sql-{cid}"},
            ),
        ]:
            store().query_write(insert(table).values(**values))
    store().query_write(
        insert(s.table__term).values(table_id="table-b", term_id="only-b")
    )
    try:
        yield
    finally:
        dispose_engine()


def test_counts_related_and_certification_are_source_scoped(
    semantic_sources: None,
) -> None:
    assert terms.count_column_attributes_by_term_id("shared", connection_id="a") == 1
    assert terms.count_column_attributes_by_term_id("shared") == 2
    assert terms.fetch_related_terms("shared", connection_id="a") == []
    assert terms.fetch_related_term_ids("shared", connection_id="b") == ["only-b"]
    assert terms.fetch_related_terms_counts(term_ids=["shared"], connection_id="b") == [
        {"term_id": "shared", "count": 1}
    ]
    assert terms.fetch_related_terms("shared", connection_id="b", skip=1, limit=1) == []
    assert (
        terms.get_full_term_by_id("shared", connection_id="a")["certification"]
        == "certified"
    )
    assert (
        terms.get_full_term_by_id("shared", connection_id="b")["certification"]
        == "partial"
    )
    assert terms.fetch_all_terms(connection_id="a")[0]["certification"] == "certified"


def test_scoped_hydration_preserves_meaning_and_pages_unique_attributes(
    semantic_sources: None,
) -> None:
    store().query_write(
        insert(s.catalog_column).values(
            id="second-a", name="second", table_id="table-a"
        )
    )
    store().query_write(
        insert(s.column__has_attribute).values(
            column_id="second-a", attribute_id="attr-a"
        )
    )
    attrs = terms.fetch_column_attributes_by_term_id(
        "shared", connection_id="a", limit=1
    )
    assert [row["id"] for row in attrs] == ["attr-a"]
    assert attrs[0]["certified"] is True
    assert attrs[0]["primary_column"]["id"] == "column-a"
    assert "marker-b" not in repr(attrs)
    assert (
        terms.fetch_column_attributes_by_term_id(
            "shared", connection_id="a", skip=1, limit=1
        )
        == []
    )
    assert terms.get_full_term_by_id("only-b", connection_id="a") is None
    scoped = terms.get_full_term_by_id("shared", connection_id="a")
    global_term = terms.get_full_term_by_id("shared")
    assert scoped["description"] == global_term["description"] == "Global meaning"
    assert scoped["synonyms"] == global_term["synonyms"] == ["Vocabulary"]
    assert scoped["table_count"] == 1
    assert global_term["table_count"] == 2


def test_scoped_count_and_paths_use_canonical_owned_attributes(
    semantic_sources: None,
) -> None:
    store().query_write(
        insert(s.column__semantic_fk).values(
            column_id="column-b", attribute_id="attr-a"
        )
    )
    store().query_write(
        insert(s.column__has_attribute).values(
            column_id="column-b", attribute_id="attr-a"
        )
    )
    assert "column-b" not in repr(
        terms.fetch_column_attributes_by_term_id("shared", connection_id="a")
    )
    store().query_write(
        s.column__has_attribute.delete().where(
            s.column__has_attribute.c.column_id == "column-a"
        )
    )
    assert terms.fetch_column_attributes_by_term_id("shared", connection_id="a") == []
    assert terms.count_column_attributes_by_term_id("shared", connection_id="a") == 0
    assert terms.fetch_column_attribute_counts(connection_id="a") == []


def test_public_http_reads_use_real_source_scoped_store(
    semantic_sources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        lambda: [
            sources.SourceDescriptor(cid, cid, "sales", "postgres", "managed")
            for cid in ("a", "b")
        ],
    )
    app = FastAPI()
    app.include_router(term_router.router, prefix="/api")
    app.include_router(sql_router.router, prefix="/api")
    with TestClient(app) as client:
        params = {"connection_id": "a", "limit": 1}
        page = client.get("/api/terms", params=params)
        assert page.status_code == 200
        assert page.json()["total"] == 1
        assert page.json()["column_attribute_counts"] == [
            {"term_id": "shared", "count": 1}
        ]
        assert page.json()["sql_attribute_counts"] == [
            {"term_id": "shared", "count": 1}
        ]
        assert page.json()["related_counts"] == [{"term_id": "shared", "count": 0}]
        assert page.json()["terms"][0]["certification"] == "certified"
        assert client.get(
            "/api/terms", params={"connection_id": "b", "limit": 1, "skip": 1}
        ).json()["related_counts"] == [{"term_id": "shared", "count": 1}]
        for path in (
            "/terms/shared",
            "/terms/shared/column-attributes",
            "/terms/shared/sql-attributes",
            "/sql-attributes/sql-attr-a",
        ):
            response = client.get("/api" + path, params=params)
            assert response.status_code == 200
            assert "marker-b" not in response.text
        for path in ("/terms/shared/column-attributes", "/terms/shared/sql-attributes"):
            assert client.get("/api" + path, params={**params, "skip": 1}).json() == {
                "data": [],
                "count": 0,
                "total": 1,
            }
        assert (
            client.get("/api/terms/shared", params=params).json()["data"][
                "related_terms"
            ]
            == []
        )
        assert client.get("/api/terms/shared").json()["data"]["table_count"] == 2
        assert client.get("/api/terms/only-b", params=params).status_code == 404
        assert (
            client.get("/api/sql-attributes/sql-attr-b", params=params).status_code
            == 404
        )
        assert (
            client.get("/api/sql-attributes/missing", params=params).status_code == 404
        )


def test_table_detail_scopes_before_count_page_and_query_hydration(
    semantic_sources: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        lambda: [
            sources.SourceDescriptor(cid, cid, "sales", "postgres", "managed")
            for cid in ("a", "b")
        ],
    )
    for table, values in [
        (s.term, {"id": "local", "name": "Z local"}),
        (s.table__term, {"term_id": "local", "table_id": "table-a"}),
        (s.column_attribute__term, {"attribute_id": "attr-b", "term_id": "only-b"}),
        (s.column__has_attribute, {"attribute_id": "attr-b", "column_id": "column-a"}),
        (s.column__semantic_fk, {"attribute_id": "attr-b", "column_id": "column-a"}),
        (s.sql_query__table, {"sql_query_id": "sql-b", "table_id": "table-a"}),
        (s.sql_query, {"id": "legacy", "sql_full_query": "SELECT 'legacy-secret'"}),
        (s.sql_query__table, {"sql_query_id": "legacy", "table_id": "table-a"}),
    ]:
        # Move the existing attribute to its B-only term to expose forged local links.
        if table is s.column_attribute__term:
            store().query_write(table.delete().where(table.c.attribute_id == "attr-b"))
        store().query_write(insert(table).values(**values))
    app = FastAPI()
    app.include_router(exploration_router.router, prefix="/api")
    with TestClient(app) as client:
        path = "/api/exploration/tables/table-a/details"
        params = {"connection_id": "a", "limit": 1, "skip": 1}
        response = client.get(path, params=params)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["terms_total"] == 2
        assert [term["id"] for term in data["terms"]] == ["local"]
        assert [query["id"] for query in data["queries"]] == ["sql-a"]
        assert "B only" not in response.text
        assert "legacy-secret" not in response.text
        assert (
            client.get(path, params={**params, "skip": 2}).json()["data"]["terms"] == []
        )
        assert client.get(path).json()["data"]["terms_total"] == 3
        assert len(client.get(path).json()["data"]["queries"]) == 3
        for table_id in ("table-b", "missing"):
            assert (
                client.get(
                    f"/api/exploration/tables/{table_id}/details", params=params
                ).status_code
                == 404
            )
        b = client.get(
            "/api/exploration/tables/table-b/details", params={"connection_id": "b"}
        )
        assert b.status_code == 200
        assert {term["id"] for term in b.json()["data"]["terms"]} == {
            "shared",
            "only-b",
        }


def test_table_detail_requires_canonical_column_but_keeps_local_semantic_fk(
    semantic_sources: None,
) -> None:
    for table, values in [
        (s.term, {"id": "attribute-term", "name": "Attribute term"}),
        (s.catalog_table, {"id": "other-a", "name": "other", "schema_id": "schema-a"}),
        (
            s.catalog_column,
            {"id": "other-column-a", "name": "value", "table_id": "other-a"},
        ),
        (
            s.column_attribute,
            {
                "id": "other-attr-a",
                "name": "Other",
                "table_id": "other-a",
                "source_column": "value",
                "term_name": "Attribute term",
                "source": "semantic",
            },
        ),
        (
            s.column_attribute__term,
            {"attribute_id": "other-attr-a", "term_id": "attribute-term"},
        ),
        (
            s.column__has_attribute,
            {"column_id": "other-column-a", "attribute_id": "other-attr-a"},
        ),
        (
            s.column__has_attribute,
            {"column_id": "column-a", "attribute_id": "other-attr-a"},
        ),
    ]:
        store().query_write(insert(table).values(**values))
    # A forged HAS_ATTRIBUTE must not substitute for a valid local semantic FK.
    assert (
        exploration.fetch_table_exploration_details("table-a", connection_id="a")[
            "terms_total"
        ]
        == 1
    )
    store().query_write(
        insert(s.column__semantic_fk).values(
            column_id="column-a", attribute_id="other-attr-a"
        )
    )
    assert (
        exploration.fetch_table_exploration_details("table-a", connection_id="a")[
            "terms_total"
        ]
        == 2
    )
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == "other-attr-a")
        .values(source_column="missing")
    )
    assert (
        exploration.fetch_table_exploration_details("table-a", connection_id="a")[
            "terms_total"
        ]
        == 1
    )
