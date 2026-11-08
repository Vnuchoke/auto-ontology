# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.dal import datasources, schema
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database,  # noqa: F401
)
from auto_ontology.server.models import DatabaseSummary


def test_catalog_query_returns_separate_display_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Mock()
    database.query_read.return_value = [
        {
            "id": "a",
            "name": "sales",
            "display_label": "Sales A",
            "owner_id": None,
            "schema_count": 1,
        }
    ]
    monkeypatch.setattr(datasources, "store", lambda: database)
    result = datasources.fetch_databases()
    assert result[0]["display_label"] == "Sales A"
    assert result[0]["name"] == "sales"
    assert "display_label" in database.query_read.call_args.args[0].selected_columns


@pytest.fixture
def labeled_catalog(request: pytest.FixtureRequest) -> Iterator[None]:
    request.getfixturevalue("identity_database")
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        for source_id, label in [("a", "Sales A"), ("b", "Sales B")]:
            store().query_write(
                schema.catalog_database.insert().values(
                    id=source_id,
                    name="sales",
                    display_label=label,
                    configuration_origin="managed",
                )
            )
            store().query_write(
                schema.catalog_schema.insert().values(
                    id=f"{source_id}-schema", database_id=source_id, name="public"
                )
            )
            store().query_write(
                schema.catalog_table.insert().values(
                    id=f"{source_id}-table",
                    schema_id=f"{source_id}-schema",
                    name="orders",
                )
            )
        yield
    finally:
        dispose_engine()


def test_catalog_labels_preserve_physical_names_and_identity_on_rename(
    labeled_catalog: None,
) -> None:
    before = {row["id"]: row for row in datasources.fetch_databases()}
    assert set(before) == {"a", "b"}
    assert {key: row["display_label"] for key, row in before.items()} == {
        "a": "Sales A",
        "b": "Sales B",
    }
    schemas_before = {
        key: datasources.fetch_schemas_for_database(key) for key in before
    }
    for source_id in before:
        assert before[source_id]["name"] == "sales"
        assert before[source_id]["num_of_schemas"] == 1
        assert datasources.fetch_databases(connection_id=source_id) == [
            before[source_id]
        ]
        assert schemas_before[source_id]["schemas"][0]["id"] == f"{source_id}-schema"

    store().query_write(
        schema.catalog_database.update()
        .where(schema.catalog_database.c.id == "a")
        .values(display_label="Sales A renamed")
    )
    after = {row["id"]: row for row in datasources.fetch_databases()}
    assert after == {
        "a": {**before["a"], "display_label": "Sales A renamed"},
        "b": before["b"],
    }
    assert {
        key: datasources.fetch_schemas_for_database(key) for key in after
    } == schemas_before


def test_catalog_http_exposes_labels_without_replacing_names(
    labeled_catalog: None,
) -> None:
    from auto_ontology.server.datasources.router import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get("/api/datasources/dbs")
    assert response.status_code == 200
    assert {
        row["id"]: (row["name"], row["display_label"])
        for row in response.json()["data"]
    } == {"a": ("sales", "Sales A"), "b": ("sales", "Sales B")}


def test_database_summary_keeps_label_and_accepts_legacy_payload() -> None:
    assert (
        DatabaseSummary(id="a", name="sales", display_label="Sales A").model_dump()[
            "display_label"
        ]
        == "Sales A"
    )
    assert DatabaseSummary(id="a", name="sales").display_label is None
