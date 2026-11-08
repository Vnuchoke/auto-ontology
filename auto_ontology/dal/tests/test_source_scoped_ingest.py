# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog-only ingest against an authorized, disposable store; no source/network."""

from __future__ import annotations

from types import SimpleNamespace
from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import select

from auto_ontology.catalog.constants import Labels
from auto_ontology.catalog.ingest import ingest_catalog
from auto_ontology.catalog.store.rows import upsert_row
from auto_ontology.catalog.store.schemas import PendingRows, merge_schema_edges
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)


@pytest.fixture
def catalog_store(identity_database: None) -> Iterator[None]:
    from alembic import command
    from alembic.config import Config

    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        yield
    finally:
        dispose_engine()


def _connector(source: dict) -> SimpleNamespace:
    table = source["table"]
    empty = pd.DataFrame
    return SimpleNamespace(
        connection_id=source["connection_id"],
        database_name=source["connection"]["database"],
        dialect="postgres",
        get_tables=lambda: pd.DataFrame(
            [{"table_schema": "public", "table_name": "orders"}]
        ),
        get_columns=lambda: pd.DataFrame(
            [
                {
                    "table_schema": "public",
                    "table_name": "orders",
                    "column_name": name,
                    "data_type": "text",
                    "is_nullable": False,
                    "ordinal_position": i + 1,
                }
                for i, name in enumerate(table["columns"])
            ]
        ),
        get_views=empty,
        get_queries=empty,
        get_pks=lambda: pd.DataFrame(
            [
                {
                    "table_schema": "public",
                    "table_name": "orders",
                    "column_name": "order_id",
                }
            ]
        ),
        get_fks=lambda: pd.DataFrame(
            [
                {
                    "table_schema": "public",
                    "table_name": "orders",
                    "column_name": "amount",
                    "referenced_schema": "public",
                    "referenced_table": "orders",
                    "referenced_column": "order_id",
                }
            ]
        ),
    )


def _register(sources: list[dict]) -> None:
    for source in sources:
        store().query_write(
            s.catalog_database.insert().values(
                id=source["connection_id"],
                name="sales",
                display_label=source["connection_label"],
                configuration_origin="managed",
                credential_ref=f"connections/{source['connection_id']}",
                connection=source["connection"],
            )
        )


def _snapshot(connection_id: str) -> dict:
    schemas = store().query_read(
        select(s.catalog_schema).where(s.catalog_schema.c.database_id == connection_id)
    )
    tables = store().query_read(
        select(s.catalog_table).where(
            s.catalog_table.c.schema_id.in_([r["id"] for r in schemas])
        )
    )
    columns = store().query_read(
        select(s.catalog_column).where(
            s.catalog_column.c.table_id.in_([r["id"] for r in tables])
        )
    )
    fks = store().query_read(
        select(s.column__foreign_key).where(
            s.column__foreign_key.c.source_column_id.in_([r["id"] for r in columns])
        )
    )
    roots = store().query_read(
        select(s.catalog_database).where(s.catalog_database.c.id == connection_id)
    )
    return {
        "roots": roots,
        "schemas": schemas,
        "tables": tables,
        "columns": columns,
        "fks": fks,
    }


def test_catalog_only_root_identity_is_explicit_not_its_physical_name(
    catalog_store: None, same_name_sources: dict
) -> None:
    ids = []
    for source in same_name_sources["sources"]:
        root_id = source["connection_id"]
        ids.append(
            upsert_row(
                Labels.DB,
                {"id": root_id},
                {
                    "id": root_id,
                    "name": "sales",
                    "display_label": source["connection_label"],
                },
            )
        )
    assert ids == [source["connection_id"] for source in same_name_sources["sources"]]
    assert ids[0] != ids[1]
    with pytest.raises(ValueError, match="connection_id"):
        upsert_row(Labels.DB, {"name": "sales"}, {"name": "sales"})


def test_same_name_ingest_and_incremental_diff_preserve_b(
    catalog_store: None, same_name_sources: dict
) -> None:
    a, b = same_name_sources["sources"]
    _register([a, b])
    for source in (a, b):
        tables, columns = ingest_catalog(_connector(source))
        snapshot = _snapshot(source["connection_id"])
        assert len(snapshot["tables"]) == 1
        assert {column["name"] for column in snapshot["columns"]} == set(
            source["table"]["columns"]
        )
        assert set(columns.column_name) == set(source["table"]["columns"])
        assert set(tables.id) == {snapshot["tables"][0]["id"]}
        for column in snapshot["columns"]:
            store().query_write(
                s.catalog_column.update()
                .where(s.catalog_column.c.id == column["id"])
                .values(
                    description=f"curated {source['connection_label']} {column['name']}",
                    sample_values=str(
                        [source["table"]["sample_rows"][0][column["name"]]]
                    ),
                )
            )
        store().query_write(
            s.catalog_table.update()
            .where(s.catalog_table.c.id == snapshot["tables"][0]["id"])
            .values(description="curated orders")
        )

    before_a = _snapshot(a["connection_id"])
    before_b = _snapshot(b["connection_id"])
    assert before_a["tables"][0]["id"] != before_b["tables"][0]["id"]
    assert "draft_status" not in {c["name"] for c in before_b["columns"]}
    assert {c["id"] for c in before_a["columns"]}.isdisjoint(
        c["id"] for c in before_b["columns"]
    )
    assert len(before_a["fks"]) == len(before_b["fks"]) == 1
    for snapshot in (before_a, before_b):
        assert snapshot["fks"][0]["target_column_id"] in {
            c["id"] for c in snapshot["columns"]
        }

    connector = _connector(a)
    original_columns = connector.get_columns
    connector.get_columns = lambda: original_columns().assign(data_type="varchar")
    tables, columns = ingest_catalog(connector)
    after_a = _snapshot(a["connection_id"])
    assert _snapshot(b["connection_id"]) == before_b
    assert set(tables.id) == {before_a["tables"][0]["id"]}
    assert set(columns.id) == {c["id"] for c in before_a["columns"]}
    assert after_a["tables"] == before_a["tables"]
    assert after_a["schemas"] == before_a["schemas"]
    for old, new in zip(
        sorted(before_a["columns"], key=lambda c: c["id"]),
        sorted(after_a["columns"], key=lambda c: c["id"]),
        strict=True,
    ):
        assert new == {**old, "data_type": "varchar"}
    assert after_a["roots"][0] == {
        **before_a["roots"][0],
        "pulled": after_a["roots"][0]["pulled"],
    }

    a["table"]["columns"] = ["order_id", "amount", "new_local_column"]
    connector = _connector(a)
    connector.get_fks = lambda: pd.DataFrame()
    connector.get_pks = lambda: pd.DataFrame()
    ingest_catalog(connector)
    changed_a = _snapshot(a["connection_id"])
    assert {c["name"] for c in changed_a["columns"]} == set(a["table"]["columns"])
    assert changed_a["tables"][0]["pk"] is None
    assert changed_a["fks"] == []
    assert _snapshot(b["connection_id"]) == before_b


def test_foreign_parent_and_source_context_are_rejected(
    catalog_store: None, same_name_sources: dict
) -> None:
    a, b = same_name_sources["sources"]
    _register([a, b])
    ingest_catalog(_connector(a))
    ingest_catalog(_connector(b))
    before = _snapshot(b["connection_id"])
    schema_b = before["schemas"][0]
    with pytest.raises(ValueError, match="(?i)(parent|source|connection)"):
        upsert_row(
            Labels.SCHEMA,
            {
                "id": schema_b["id"],
                "connection_id": a["connection_id"],
                "name": "public",
            },
            {"id": schema_b["id"], "name": "public"},
            parent_id=a["connection_id"],
        )
    table_b = before["tables"][0]
    with pytest.raises(ValueError, match="(?i)(parent|source|connection)"):
        upsert_row(
            Labels.TABLE,
            {"id": table_b["id"]},
            {"id": table_b["id"], "name": "orders"},
            parent_id=_snapshot(a["connection_id"])["schemas"][0]["id"],
        )
    with pytest.raises(ValueError, match="(?i)(parent|source)"):
        merge_schema_edges(
            [
                {
                    "uid": table_b["id"],
                    "vid": _snapshot(a["connection_id"])["schemas"][0]["id"],
                }
            ],
            Labels.SCHEMA,
            Labels.TABLE,
            PendingRows(),
        )
    with pytest.raises(ValueError, match="(?i)(source|connection)"):
        upsert_row(
            Labels.DB,
            {"id": a["connection_id"]},
            {"id": b["connection_id"], "name": "sales"},
        )
    connector = _connector(a)
    original_tables = connector.get_tables
    connector.get_tables = lambda: original_tables().assign(id=table_b["id"])
    with pytest.raises(ValueError, match="(?i)(source|parent)"):
        ingest_catalog(connector)
    connector = _connector(a)
    original_columns = connector.get_columns
    connector.get_columns = lambda: original_columns().assign(
        id=before["columns"][0]["id"]
    )
    with pytest.raises(ValueError, match="(?i)(source|parent)"):
        ingest_catalog(connector)
    connector = _connector(a)
    connector.database_name = "wrong_physical_name"
    with pytest.raises(ValueError, match="(?i)(name|source|connection)"):
        ingest_catalog(connector)
    connector = _connector(a)
    connector.connection_id = "unknown-source"
    with pytest.raises(ValueError, match="(?i)(source|connection|registered)"):
        ingest_catalog(connector)
    assert _snapshot(b["connection_id"]) == before
