# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata source discovery and atomic environment registration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
import asyncio
import importlib
from typing import Any
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config

from auto_ontology.connectors import sources
from auto_ontology.dal import connections
from auto_ontology.dal.session import StoreConn, dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import identity_database  # noqa: F401


@pytest.fixture
def registered_store(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    dispose_engine()
    request.getfixturevalue("identity_database")
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    command.upgrade(Config("alembic.ini"), "head")
    yield
    dispose_engine()


def url(name: str = "sales", *, source_id: str | None = None, label: str = "") -> str:
    result = f"postgresql://synthetic:secret@invalid/{name}?sslmode=require"
    if source_id:
        result += f"&metadata_connection_id={source_id}"
    if label:
        result += f"&metadata_label={label}"
    return result


@pytest.mark.parametrize(
    "raw,name",
    [
        ("sqlite:////tmp/sales.sqlite", "sales"),
        ("duckdb:///tmp/sales.duckdb", "sales"),
        ("duckdb://:memory:", "memory"),
        ("snowflake://u:p@invalid?database=SALES&warehouse=WH", "SALES"),
    ],
)
def test_physical_names_without_driver(raw: str, name: str) -> None:
    assert sources.parse_env_sources(raw)[0].database_name == name


def test_reserved_options_are_private_and_removed() -> None:
    source = sources.parse_env_sources(
        url(source_id=str(uuid4()), label="Sales+local")
    )[0]
    assert source.display_label == "Sales local"
    assert "secret" not in repr(source)
    assert "metadata_" not in source._driver_url
    assert "sslmode=require" in source._driver_url
    assert source.dialect == "postgres"


def test_postgres_query_database_overrides_path() -> None:
    assert (
        sources.parse_env_sources(url() + "&dbname=inventory")[0].database_name
        == "inventory"
    )


def test_invalid_postgres_option_is_safe() -> None:
    with pytest.raises(ValueError, match="invalid_source_selector") as error:
        sources.parse_env_sources(url() + "&unknown=secret")
    assert "secret" not in str(error.value)


def test_blank_env_needs_no_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTION_STRINGS", " \n")

    def forbidden() -> None:
        pytest.fail("blank env attempted store access")

    monkeypatch.setattr(connections, "list_source_metadata", forbidden)
    assert sources.bootstrap_sources() == []


def test_duplicate_explicit_ids_rejected() -> None:
    source_id = str(uuid4())
    with pytest.raises(ValueError, match="invalid_source_selector"):
        sources.parse_env_sources(
            url(source_id=source_id, label="A")
            + ","
            + url("other", source_id=source_id, label="B")
        )


@pytest.mark.parametrize("scheme", ["sqlite", "duckdb"])
def test_file_driver_url_preserves_path_and_nonmetadata_options(scheme: str) -> None:
    raw = f"{scheme}:///tmp/sales.db?mode=ro&metadata_label=Local&opt=a%2Bb"
    parsed = sources.parse_env_sources(raw)[0]
    assert parsed._driver_url == f"{scheme}:///tmp/sales.db?mode=ro&opt=a%2Bb"


@pytest.mark.parametrize(
    "raw",
    [
        url() + "&metadata_label=",
        url() + "&metadata_connection_id=no",
        url() + "&metadata_label=A&metadata_label=B",
        url() + "," + url(),
        "unsupported://u:secret@invalid/sales",
        "postgresql://u:secret@invalid/",
    ],
)
def test_invalid_batch_is_safe(raw: str) -> None:
    with pytest.raises(ValueError) as error:
        sources.parse_env_sources(raw)
    assert "secret" not in str(error.value)


@pytest.mark.usefixtures("registered_store")
def test_plain_reboot_reorder_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTION_STRINGS", url() + "," + url("inventory"))
    first = sources.bootstrap_sources()
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        url("inventory") + "," + url().replace("secret", "rotated"),
    )
    assert set(sources.bootstrap_sources()) == set(first)
    assert set(sources.list_active_sources()) == set(first)
    assert len(connections.list_source_metadata()) == 2


@pytest.mark.usefixtures("registered_store")
def test_explicit_same_names_and_bare_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [str(uuid4()), str(uuid4())]
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        ",".join(
            url(source_id=i, label=label) for i, label in zip(ids, ["Local", "Prod"])
        ),
    )
    assert {d.connection_id for d in sources.bootstrap_sources()} == set(ids)
    monkeypatch.setenv("CONNECTION_STRINGS", url())
    with pytest.raises(ValueError):
        sources.bootstrap_sources()


@pytest.mark.usefixtures("registered_store")
def test_managed_preserved_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    managed_id = str(uuid4())
    store().query_write(
        "INSERT INTO catalog_database (id,name,display_label,configuration_origin,dialect) VALUES (:id,'sales','Managed','managed','postgres')",
        {"id": managed_id},
    )
    monkeypatch.setenv("CONNECTION_STRINGS", url(source_id=managed_id, label="Stolen"))
    with pytest.raises(ValueError):
        sources.bootstrap_sources()
    monkeypatch.setenv("CONNECTION_STRINGS", url())
    with pytest.raises(ValueError):
        sources.bootstrap_sources()
    monkeypatch.setenv("CONNECTION_STRINGS", url(source_id=str(uuid4()), label="Env"))
    assert sources.bootstrap_sources()[0].display_label == "Env"
    monkeypatch.setenv("CONNECTION_STRINGS", " ")
    assert [s.connection_id for s in sources.list_active_sources()] == [managed_id]
    assert sources.list_active_sources()[0].dialect == "postgres"


@pytest.mark.usefixtures("registered_store")
def test_whole_batch_label_conflict_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        url(source_id=str(uuid4()), label="Sales")
        + ","
        + url("other", source_id=str(uuid4()), label="+sALES+"),
    )
    with pytest.raises(ValueError):
        sources.bootstrap_sources()
    assert connections.list_source_metadata() == []


@pytest.mark.usefixtures("registered_store")
def test_read_only_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTION_STRINGS", url())
    with pytest.raises(ValueError, match="source_not_ready"):
        sources.list_active_sources()
    expected = sources.bootstrap_sources()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("discovery attempted credentials, driver, or write")

    monkeypatch.setattr(connections, "read_secret", forbidden)
    monkeypatch.setattr(connections, "list_connections", forbidden)
    monkeypatch.setattr(type(store()), "query_write", forbidden)
    query_read = type(store()).query_read

    def metadata_only(
        handle: StoreConn, statement: Any, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if hasattr(statement, "selected_columns"):
            assert not {"connection", "credential_ref"} & set(
                statement.selected_columns.keys()
            )
        return query_read(handle, statement, parameters)

    monkeypatch.setattr(type(store()), "query_read", metadata_only)
    from auto_ontology.connectors import registry

    monkeypatch.setattr(registry, "create_connector", forbidden)
    assert sources.list_active_sources() == expected
    assert sources.list_active_sources() == expected


@pytest.mark.usefixtures("registered_store")
def test_concurrent_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTION_STRINGS", url() + "," + url("inventory"))
    # Create the cached engine on the main thread, not concurrently.
    connections.list_source_metadata()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: sources.bootstrap_sources(), range(2)))
    assert results[0] == results[1]
    assert len(connections.list_source_metadata()) == 2


@pytest.mark.usefixtures("registered_store")
def test_explicit_rename_keeps_identity_and_stores_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = str(uuid4())
    monkeypatch.setenv("CONNECTION_STRINGS", url(source_id=source_id, label="Before"))
    sources.bootstrap_sources()
    monkeypatch.setenv("CONNECTION_STRINGS", url(source_id=source_id, label="After"))
    with pytest.raises(ValueError, match="source_not_ready"):
        sources.list_active_sources()
    assert sources.bootstrap_sources()[0].connection_id == source_id
    row = store().query_read(
        "SELECT name, display_label, connection, credential_ref FROM catalog_database"
    )[0]
    assert row == {
        "name": "sales",
        "display_label": "After",
        "connection": None,
        "credential_ref": None,
    }
    monkeypatch.setenv(
        "CONNECTION_STRINGS", url("other", source_id=source_id, label="After")
    )
    with pytest.raises(ValueError):
        sources.bootstrap_sources()
    assert connections.list_source_metadata()[0]["name"] == "sales"


@pytest.mark.usefixtures("registered_store")
def test_postgres_normalization_is_not_casefold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        url(source_id=str(uuid4()), label="Stra%C3%9Fe")
        + ","
        + url("other", source_id=str(uuid4()), label="STRASSE"),
    )
    assert len(sources.bootstrap_sources()) == 2


@pytest.mark.usefixtures("registered_store")
@pytest.mark.parametrize("origin", [None, "managed"])
def test_late_conflict_rolls_back_whole_batch(
    origin: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_id = str(uuid4())
    store().query_write(
        "INSERT INTO catalog_database (id,name,display_label,configuration_origin) VALUES (:id,'sales','Reserved',:origin)",
        {"id": source_id, "origin": origin},
    )
    before = connections.list_source_metadata()
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        url("new") + "," + url(source_id=source_id, label="Changed"),
    )
    with pytest.raises(ValueError):
        sources.bootstrap_sources()
    assert connections.list_source_metadata() == before


@pytest.mark.usefixtures("registered_store")
def test_existing_global_label_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    store().query_write(
        "INSERT INTO catalog_database (name,display_label) VALUES ('inactive','Reserved')"
    )
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        url("new") + "," + url(source_id=str(uuid4()), label="+RESERVED+"),
    )
    with pytest.raises(ValueError, match="source_conflict"):
        sources.bootstrap_sources()
    assert len(connections.list_source_metadata()) == 1


@pytest.mark.parametrize(
    "module_name",
    ["auto_ontology.server.__main__", "auto_ontology.ingestion_service.__main__"],
)
def test_lifespan_registration_order_and_failure(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(module_name)
    calls: list[str] = []
    monkeypatch.setattr(
        module, "require_current_schema", lambda: calls.append("schema")
    )

    def fail_registration() -> None:
        calls.append("bootstrap")
        raise ValueError("source_not_ready")

    monkeypatch.setattr(module, "bootstrap_sources", fail_registration, raising=False)
    if hasattr(module, "create_app"):
        app = module.create_app()
        app.openapi()
        assert calls == []
        monkeypatch.setattr(module, "get_pool", lambda: calls.append("workers"))
    else:
        app = module.app
        monkeypatch.setattr(module, "DataScheduler", lambda: calls.append("scheduler"))

    async def start() -> None:
        async with module.lifespan(app):
            pytest.fail("invalid sources reached readiness")

    with pytest.raises(ValueError, match="source_not_ready"):
        asyncio.run(start())
    assert calls == ["schema", "bootstrap"]
