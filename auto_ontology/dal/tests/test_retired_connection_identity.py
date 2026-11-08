# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retired IDs stay unavailable even after their catalog root disappears."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.catalog.store.rows import upsert_row
from auto_ontology.dal import connections, schema
from auto_ontology.dal.session import dispose_engine, store, write_transaction
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)


@pytest.fixture
def retired_database(identity_database: None) -> Iterator[None]:
    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    try:
        yield
    finally:
        dispose_engine()


@pytest.mark.parametrize(
    "writer", ["env", "managed", "catalog", "catalog-list", "catalog-tuple"]
)
def test_deleted_root_cannot_be_resurrected(
    retired_database: None, writer: str
) -> None:
    source_id = str(uuid4())
    connections.insert_connection(
        connection_id=source_id,
        display_label="A",
        database_name="sales",
        connection="{}",
    )
    store().query_write(
        schema.catalog_database.delete().where(
            schema.catalog_database.c.id == source_id
        )
    )
    store().query_write(
        "INSERT INTO retired_connection VALUES (:id)", {"id": source_id}
    )
    with pytest.raises(
        ValueError, match="^source_retired: connection_id cannot be reused$"
    ):
        if writer == "env":
            connections.register_env_source(
                connection_id=source_id, display_label="A", database_name="sales"
            )
        elif writer == "managed":
            connections.insert_connection(
                connection_id=source_id,
                display_label="A",
                database_name="sales",
                connection="{}",
            )
        else:
            upsert_row(
                {
                    "catalog": "Database",
                    "catalog-list": ["Database"],
                    "catalog-tuple": ("Database",),
                }[writer],
                {"id": source_id},
                {"id": source_id, "name": "sales", "display_label": "A"},
            )


def test_finalization_frees_label_but_never_identity(retired_database: None) -> None:
    a, b, c = (str(uuid4()) for _ in range(3))
    for source_id, label in ((a, "A"), (b, "B")):
        connections.insert_connection(
            connection_id=source_id,
            display_label=label,
            database_name="sales",
            connection="{}",
        )
    with pytest.raises(ValueError, match="source_conflict"):
        connections.finalize_connection_removal(a)
    connections.clear_connection(connection_id=a)
    connections.finalize_connection_removal(a)
    connections.insert_connection(
        connection_id=c, display_label="A", database_name="sales", connection="{}"
    )
    connections.finalize_connection_removal(a)
    assert {row["id"] for row in connections.list_source_metadata()} == {b, c}
    with pytest.raises(ValueError, match="source_retired"):
        connections.assert_connection_id_available(a)


@pytest.mark.parametrize(
    "binding", ["configuration_origin", "connection", "credential_ref"]
)
def test_finalization_refuses_any_remaining_binding(
    retired_database: None, binding: str
) -> None:
    source_id = str(uuid4())
    values = {
        "id": source_id,
        "name": "sales",
        "display_label": "A",
        binding: {} if binding == "connection" else "managed",
    }
    store().query_write(schema.catalog_database.insert().values(**values))
    with pytest.raises(ValueError, match="source_conflict"):
        connections.finalize_connection_removal(source_id)
    assert store().query_read("SELECT id FROM catalog_database") == [{"id": source_id}]
    assert not store().query_read("SELECT * FROM retired_connection")


def test_finalization_rolls_back_root_and_tombstone(retired_database: None) -> None:
    source_id = str(uuid4())
    store().query_write(
        schema.catalog_database.insert().values(
            id=source_id, name="sales", display_label="A"
        )
    )
    with pytest.raises(RuntimeError, match="rollback"):
        with write_transaction():
            connections.finalize_connection_removal(source_id)
            raise RuntimeError("rollback")
    assert store().query_read("SELECT id FROM catalog_database") == [{"id": source_id}]
    assert not store().query_read("SELECT * FROM retired_connection")


def test_missing_root_is_retired_idempotently(retired_database: None) -> None:
    source_id = str(uuid4())
    connections.finalize_connection_removal(source_id)
    connections.finalize_connection_removal(source_id)
    assert store().query_read("SELECT * FROM retired_connection") == [
        {"connection_id": source_id}
    ]


def test_global_reset_preserves_tombstones_and_active_roots(
    retired_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal import reset

    retired_id, active_id = str(uuid4()), str(uuid4())
    connections.finalize_connection_removal(retired_id)
    connections.insert_connection(
        connection_id=active_id,
        display_label="A",
        database_name="sales",
        connection="{}",
    )
    monkeypatch.setattr(reset, "_delete_vectors", lambda *args, **kwargs: 0)
    reset.delete_all_data()
    assert store().query_read("SELECT * FROM retired_connection") == [
        {"connection_id": retired_id}
    ]
    assert connections.list_source_metadata()[0]["id"] == active_id


def test_model_import_cannot_resurrect_retired_root(retired_database: None) -> None:
    from auto_ontology.dal.model_interchange import (
        ModelImportValidationError,
        _resolve_entities_batch,
    )

    source_id = str(uuid4())
    connections.finalize_connection_removal(source_id)
    with pytest.raises(ModelImportValidationError, match="source_retired"):
        with write_transaction():
            _resolve_entities_batch(
                schema.catalog_database,
                [(source_id, {"id": source_id, "name": "sales", "display_label": "A"})],
            )
    assert not connections.list_source_metadata()


def test_bootstrap_cannot_reuse_retired_explicit_id(
    retired_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.connectors.sources import bootstrap_sources, list_active_sources

    source_id = str(uuid4())
    connections.finalize_connection_removal(source_id)
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        f"postgresql://user:password@localhost/sales?metadata_connection_id={source_id}&metadata_label=A",
    )
    with pytest.raises(ValueError, match="source_not_ready"):
        list_active_sources()
    with pytest.raises(ValueError, match="source_retired"):
        bootstrap_sources()
    assert not connections.list_source_metadata()


def test_finalization_serializes_with_catalog_root_creation(
    retired_database: None,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    source_id = str(uuid4())
    start = Barrier(2)

    def finalize() -> None:
        start.wait(timeout=10)
        connections.finalize_connection_removal(source_id)

    def register() -> None:
        start.wait(timeout=10)
        try:
            upsert_row(
                "Database",
                {"id": source_id},
                {"id": source_id, "name": "sales", "display_label": "A"},
            )
        except ValueError as exc:
            assert str(exc) == "source_retired: connection_id cannot be reused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(finalize), executor.submit(register)]
        for future in futures:
            future.result(timeout=15)
    assert not connections.list_source_metadata()
    with pytest.raises(ValueError, match="source_retired"):
        connections.assert_connection_id_available(source_id)


def test_upgrade_from_previous_head_preserves_existing_ids(
    identity_database: None,
) -> None:
    from sqlalchemy import create_engine, inspect, text
    from auto_ontology.dal.session import sqlalchemy_url

    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "d27b94f58c31")
    engine = create_engine(sqlalchemy_url())
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO catalog_database (id, name, display_label) VALUES ('historical', 'sales', 'A')"
                )
            )
        command.upgrade(config, "head")
        with engine.connect() as conn:
            assert conn.execute(
                text("SELECT id FROM catalog_database")
            ).scalars().all() == ["historical"]
            assert conn.execute(text("SELECT * FROM retired_connection")).all() == []
            assert not inspect(conn).get_foreign_keys("retired_connection")
            assert [
                column["name"]
                for column in inspect(conn).get_columns("retired_connection")
            ] == ["connection_id"]
        command.check(config)
        command.downgrade(config, "d27b94f58c31")
        with engine.connect() as conn:
            assert "retired_connection" not in inspect(conn).get_table_names()
            assert conn.execute(
                text("SELECT id FROM catalog_database")
            ).scalars().all() == ["historical"]
    finally:
        engine.dispose()


def test_downgrade_cannot_discard_retired_identities(retired_database: None) -> None:
    source_id = str(uuid4())
    connections.finalize_connection_removal(source_id)
    with pytest.raises(RuntimeError, match="coordinated backup"):
        command.downgrade(Config(str(ROOT / "alembic.ini")), "d27b94f58c31")
    assert store().query_read("SELECT * FROM retired_connection") == [
        {"connection_id": source_id}
    ]
    assert store().query_read("SELECT version_num FROM alembic_version") == [
        {"version_num": "e38c06a17f42"}
    ]
