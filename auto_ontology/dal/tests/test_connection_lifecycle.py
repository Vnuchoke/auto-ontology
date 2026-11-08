# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle writes use root IDs and PostgreSQL's final label constraint."""

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.dal import connections, session
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)
from auto_ontology.server.connections import service


@pytest.fixture(autouse=True)
def isolated_engine() -> Iterator[None]:
    session.dispose_engine()
    try:
        yield
    finally:
        session.dispose_engine()


@pytest.fixture
def lifecycle_database(
    identity_database: None, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    try:
        yield
    finally:
        session.dispose_engine()


def test_selected_write_and_detach_preserve_same_named_neighbor(
    lifecycle_database: None,
) -> None:
    for identity in ("one", "two"):
        connections.insert_connection(
            connection_id=identity,
            database_name="sales",
            display_label=identity,
            connection=json.dumps(
                {"type": "postgresql", "database": "sales", "password": identity}
            ),
            credential_ref="legacy/exact" if identity == "one" else None,
        )
    connections.clear_connection(connection_id="one")
    rows = {
        row["id"]: row
        for row in session.store().query_read("SELECT * FROM catalog_database")
    }
    assert rows["one"]["configuration_origin"] is None
    assert rows["one"]["credential_ref"] is None
    assert rows["one"]["connection"] is None
    assert rows["two"]["configuration_origin"] == "managed"
    assert rows["two"]["connection"]["password"] == "two"
    assert rows["one"]["name"] == rows["two"]["name"] == "sales"


@pytest.mark.parametrize("origin", [None, "env"])
def test_insert_cannot_claim_another_origins_id(
    lifecycle_database: None, origin: str | None
) -> None:
    session.store().query_write(
        "INSERT INTO catalog_database (id, name, display_label, configuration_origin) VALUES ('existing', 'sales', 'existing', :origin)",
        {"origin": origin},
    )
    with pytest.raises(ValueError, match="source_conflict"):
        connections.insert_connection(
            connection_id="existing",
            database_name="sales",
            display_label="replacement",
            connection='{"type":"postgresql"}',
        )
    assert connections.list_source_metadata()[0]["configuration_origin"] == origin
    assert connections.list_source_metadata()[0]["display_label"] == "existing"


def test_concurrent_db_label_conflict_compensates_only_loser_path(
    lifecycle_database: None,
) -> None:
    barrier = Barrier(2)
    original_insert = connections.insert_connection
    paths: set[str] = set()

    def simultaneous_insert(**kwargs: Any) -> dict[str, Any]:
        barrier.wait(timeout=10)
        return original_insert(**kwargs)

    def create(label: str) -> dict[str, Any] | service.LabelConflictError:
        try:
            return service.create_connection(
                {"type": "postgresql", "database": "sales"}, display_label=label
            )
        except service.LabelConflictError as exc:
            return exc

    with (
        patch.object(service, "insert_connection", side_effect=simultaneous_insert),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(
            service, "write_secret", side_effect=lambda path, _: paths.add(path)
        ) as write,
        patch.object(service, "delete_secrets", side_effect=paths.remove) as delete,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as ingest,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(pool.map(create, [" Sales ", "sales"]))
    winners = [result for result in results if isinstance(result, dict)]
    assert len(winners) == 1
    assert (
        sum(isinstance(result, service.LabelConflictError) for result in results) == 1
    )
    winner_path = f"connections/{winners[0]['connection_id']}"
    assert paths == {winner_path}
    assert write.call_count == 2
    assert delete.call_count == 1
    assert delete.call_args.args[0] != winner_path
    assert ingest.call_count == 1
    rows = connections.list_source_metadata()
    assert len(rows) == 1 and rows[0]["id"] == winners[0]["connection_id"]


def test_database_normalized_label_preflight_never_writes_vault(
    lifecycle_database: None,
) -> None:
    connections.insert_connection(
        connection_id="existing",
        database_name="sales",
        display_label="SaLeS",
        connection='{"type":"postgresql"}',
    )
    with (
        patch.object(service, "write_secret") as write,
        patch.object(service, "is_vault_configured", return_value=True),
    ):
        with pytest.raises(service.LabelConflictError):
            service.create_connection(
                {"type": "postgresql", "database": "sales"}, display_label=" sales "
            )
    write.assert_not_called()


def test_new_same_named_source_does_not_touch_existing_legacy_credentials(
    lifecycle_database: None,
) -> None:
    connections.insert_connection(
        connection_id="old-id",
        database_name="sales",
        display_label="old",
        connection="",
        credential_ref="legacy/exact/sales",
    )
    driver = {"type": "postgresql", "database": "sales", "password": "new-synthetic"}
    with (
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(
            connections,
            "read_secret",
            return_value={
                "type": "postgresql",
                "database": "sales",
                "password": "old-synthetic",
            },
        ) as read,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        new = service.create_connection(driver, display_label="new")
        read.assert_not_called()
        write.assert_called_once_with(f"connections/{new['connection_id']}", driver)
        assert (
            connections.load_source_connection("old-id")["password"] == "old-synthetic"
        )
        read.assert_called_once_with("legacy/exact/sales")
    assert len(connections.list_source_metadata()) == 2
    assert (
        connections.get_connection_storage("old-id")["credential_ref"]
        == "legacy/exact/sales"
    )
