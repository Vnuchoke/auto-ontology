# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
import json
from unittest.mock import Mock
from typing import Any

import pandas as pd
import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.dal import schema as s, terms
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)
from auto_ontology.semantic import embed


@pytest.fixture
def projections(identity_database: None) -> Iterator[None]:
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        store().query_write(
            s.term.insert().values(id="shared", name="Order", source="semantic")
        )
        for cid in ("a", "b"):
            for table, values in (
                (s.catalog_database, dict(id=cid, name="sales", display_label=cid)),
                (
                    s.catalog_schema,
                    dict(id=f"s-{cid}", database_id=cid, name=f"schema_{cid}"),
                ),
                (
                    s.catalog_table,
                    dict(id=f"t-{cid}", schema_id=f"s-{cid}", name="orders"),
                ),
                (
                    s.catalog_column,
                    dict(
                        id=f"c-{cid}",
                        table_id=f"t-{cid}",
                        name="status",
                        sample_values=json.dumps([f"{cid}_marker"]),
                    ),
                ),
                (s.table__term, dict(table_id=f"t-{cid}", term_id="shared")),
                (
                    s.column_attribute,
                    dict(
                        id=f"attr-{cid}",
                        name="Status",
                        term_name="Order",
                        source_column="status",
                        table_id=f"t-{cid}",
                        source="semantic",
                    ),
                ),
                (
                    s.column_attribute__term,
                    dict(attribute_id=f"attr-{cid}", term_id="shared"),
                ),
                (
                    s.column__has_attribute,
                    dict(attribute_id=f"attr-{cid}", column_id=f"c-{cid}"),
                ),
            ):
                store().query_write(table.insert().values(**values))
        yield
    finally:
        dispose_engine()


@pytest.mark.usefixtures("projections")
def test_bulk_filters_before_embedding_text_and_keeps_original_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []
    written = {}

    def batch(rows: list[dict[str, Any]], params: Any) -> pd.DataFrame:
        captured.extend(rows)
        return pd.DataFrame(
            [{**r, "metadata": {**r["metadata"], "embedding": [1.0]}} for r in rows]
        )

    def write(vdb: Any, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            written[(row["metadata"]["connection_id"], row["metadata"]["id"])] = row

    monkeypatch.setattr(embed, "batch_embed", batch)
    monkeypatch.setattr(embed, "write_embedded_rows", write)
    for cid in ("a", "b", "a"):
        captured.clear()
        embedder = embed.SemanticEmbedder(
            database_name="sales", connection_id=cid, embed_params=Mock(), vdb=Mock()
        )
        assert embed.embed_all_semantic_nodes(embedder) == 2
        text = " ".join(r["text"] for r in captured)
        assert f"{cid}_marker" in text
        assert f"{'b' if cid == 'a' else 'a'}_marker" not in text
        term = next(r for r in captured if r["metadata"]["id"] == "shared")
        assert term["metadata"]["schema_names"] == [f"schema_{cid}"]
        assert term["metadata"]["content_metadata"]["connection_id"] == cid
    assert len(written) == 4


@pytest.mark.usefixtures("projections")
def test_source_projection_is_not_the_global_glossary() -> None:
    for cid in ("a", "b"):
        projected, attrs = terms.fetch_all_terms_and_attributes(connection_id=cid)
        assert [t["id"] for t in projected] == ["shared"]
        assert projected[0]["connection_id"] == cid
        assert [a["id"] for a in attrs] == [f"attr-{cid}"]
        assert attrs[0]["connection_id"] == cid
    with pytest.raises(ValueError, match="connection_id"):
        terms.fetch_all_terms_and_attributes(connection_id="")


@pytest.mark.usefixtures("projections")
def test_attribute_and_sql_only_membership_and_foreign_links() -> None:
    store().query_write(s.table__term.delete())
    assert (
        terms.fetch_all_terms_and_attributes(connection_id="a")[0][0]["id"] == "shared"
    )
    # An attribute owned by A cannot obtain samples from a foreign bound column.
    store().query_write(
        s.column__has_attribute.insert().values(attribute_id="attr-a", column_id="c-b")
    )
    _, attrs = terms.fetch_all_terms_and_attributes(connection_id="a")
    assert len(attrs) == 1 and "b_marker" not in str(attrs)
    store().query_write(s.column_attribute__term.delete())
    for cid in ("a", "b", None):
        suffix = cid or "legacy"
        store().query_write(
            s.term.insert().values(
                id=f"sql-{suffix}", name=f"SQL {suffix}", source="semantic"
            )
        )
        store().query_write(
            s.sql_attribute.insert().values(
                id=f"sa-{suffix}", name="Total", connection_id=cid
            )
        )
        store().query_write(
            s.sql_attribute__term.insert().values(
                attribute_id=f"sa-{suffix}", term_id=f"sql-{suffix}"
            )
        )
        store().query_write(
            s.sql_query.insert().values(
                id=f"q-{suffix}", sql_full_query="SELECT 1", connection_id=cid
            )
        )
        store().query_write(
            s.sql_attribute__sql.insert().values(
                attribute_id=f"sa-{suffix}", sql_query_id=f"q-{suffix}"
            )
        )
    store().query_write(
        s.sql_query__column.insert().values(sql_query_id="q-a", column_id="c-a")
    )
    projected, attrs = terms.fetch_all_terms_and_attributes(connection_id="a")
    assert [t["id"] for t in projected] == ["sql-a"]
    assert projected[0]["schema_names"] == ["schema_a"]
    assert attrs == []
    store().query_write(
        s.sql_query__column.insert().values(sql_query_id="q-a", column_id="c-b")
    )
    assert terms.fetch_all_terms_and_attributes(connection_id="a") == ([], [])
    assert [
        t["id"] for t in terms.fetch_all_terms_and_attributes(connection_id="b")[0]
    ] == ["sql-b"]


@pytest.mark.usefixtures("projections")
def test_inline_dal_contexts_carry_verified_source_ownership() -> None:
    term_rows, attrs = terms.fetch_terms_and_attributes_for_table("t-a")
    assert term_rows[0]["connection_id"] == attrs[0]["connection_id"] == "a"
    contexts = terms.fetch_column_attribute_embedding_contexts_by_column_id("c-a")
    term, attrs = contexts[0]
    assert term["connection_id"] == attrs[0]["connection_id"] == "a"
    assert attrs[0]["table_id"] == "t-a"
    assert len(embed._build_rows("sales", term, attrs, connection_id="a")) == 2


@pytest.mark.usefixtures("projections")
def test_real_vector_projections_are_stable_and_reset_is_source_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    from auto_ontology.infra.postgres import get_postgres_connection_string
    from auto_ontology.vdb import get_semantic_vdb
    from auto_ontology.vdb.records import projection_id

    def batch(items: list[dict[str, Any]], params: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {**r, "metadata": {**r["metadata"], "embedding": [1.0, 0.0, 0.0]}}
                for r in items
            ]
        )

    def snapshot() -> list[tuple[Any, ...]]:
        with psycopg.connect(get_postgres_connection_string()) as conn:
            return conn.execute(
                "SELECT langchain_id::text, connection_id, content, embedding::text FROM vdb.semantic_layer ORDER BY connection_id, langchain_id"
            ).fetchall()

    monkeypatch.setattr(embed, "batch_embed", batch)
    for cid in ("a", "b", "a"):
        vdb = get_semantic_vdb(connection_id=cid, database_name="sales")
        try:
            assert (
                embed.embed_all_semantic_nodes(
                    embed.SemanticEmbedder("sales", cid, Mock(), vdb)
                )
                == 2
            )
        finally:
            vdb.close()
    before = snapshot()
    assert len(before) == 4
    assert {r[0] for r in before} == {
        projection_id("semantic_layer", cid, node)
        for cid in ("a", "b")
        for node in ("shared", f"attr-{cid}")
    }
    vdb = get_semantic_vdb(connection_id="a", database_name="sales", reset=True)
    vdb.close()
    assert snapshot() == [r for r in before if r[1] == "b"]


@pytest.mark.usefixtures("projections")
def test_global_edit_refreshes_each_projection_but_source_edit_only_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.server.terms import service

    rows = {}
    writes = []
    vdb = Mock()
    vdb.projection_connection_ids.return_value = ["a", "b", "removed"]

    def batch(items: list[dict[str, Any]], params: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [{**r, "metadata": {**r["metadata"], "embedding": [1.0]}} for r in items]
        )

    def write(db: Any, items: list[dict[str, Any]]) -> None:
        for r in items:
            key = (r["metadata"]["connection_id"], r["metadata"]["id"])
            rows[key] = r
            writes.append(key)

    def build(
        database_name: str, *, connection_id: str, reset: bool
    ) -> embed.SemanticEmbedder:
        return embed.SemanticEmbedder(database_name, connection_id, Mock(), vdb)

    monkeypatch.setattr(embed, "batch_embed", batch)
    monkeypatch.setattr(embed, "write_embedded_rows", write)
    monkeypatch.setattr(service, "build_semantic_embedder", build)
    monkeypatch.setattr(service, "get_semantic_vdb", lambda: vdb, raising=False)
    monkeypatch.setattr(service, "get_embed_params", Mock())
    service.refresh_term_embeddings("shared", refresh_dependent_attrs=True)
    assert rows[("a", "shared")]["metadata"]["schema_names"] == ["schema_a"]
    assert "a_marker" in rows[("a", "attr-a")]["text"]
    assert "b_marker" not in rows[("a", "attr-a")]["text"]
    vdb.delete_by_id.assert_any_call("shared", connection_id="removed")
    before_b = {key: value for key, value in rows.items() if key[0] == "b"}
    writes.clear()
    service.refresh_term_embeddings(
        "shared", refresh_dependent_attrs=True, connection_id="a"
    )
    assert all(key[0] == "a" for key in writes)
    assert {key: value for key, value in rows.items() if key[0] == "b"} == before_b
    writes.clear()
    service.update_column_attribute("shared", "attr-a", description="A context edit")
    assert writes == [("a", "attr-a")]
    assert "A context edit" in rows[("a", "attr-a")]["text"]
    assert {key: value for key, value in rows.items() if key[0] == "b"} == before_b
    terms.update_term("shared", name="Purchase", description="Global edit")
    store().query_write(
        s.term.update()
        .where(s.term.c.id == "shared")
        .values(synonyms=["Global synonym"])
    )
    service.refresh_term_embeddings("shared", refresh_dependent_attrs=True)
    assert all("Purchase" in rows[(cid, f"attr-{cid}")]["text"] for cid in ("a", "b"))
    assert all("Global edit" in rows[(cid, "shared")]["text"] for cid in ("a", "b"))
    assert all(
        "Global synonym" in rows[(cid, f"attr-{cid}")]["text"] for cid in ("a", "b")
    )
