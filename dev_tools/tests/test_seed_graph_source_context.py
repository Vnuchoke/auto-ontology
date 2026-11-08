# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The explicit fixture startup and SQL seed preserve registered source IDs."""

from unittest.mock import Mock

import pytest

from auto_ontology.connectors.sources import SourceDescriptor
from dev_tools.fixtures import seed_graph_fixture as seed


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[SourceDescriptor]:
    monkeypatch.setenv(
        "CONNECTION_STRINGS", "postgresql://fixture/pagila,sqlite:///chinook.sqlite"
    )
    descriptors = [
        SourceDescriptor("source-a", "Films", "pagila", "postgres", "env"),
        SourceDescriptor("source-b", "Music", "chinook", "sqlite", "env"),
    ]
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources", lambda: descriptors
    )
    return descriptors


def test_catalog_startup_checks_schema_and_bootstraps_before_opening_by_id(
    monkeypatch: pytest.MonkeyPatch, sources: list[SourceDescriptor]
) -> None:
    events = []
    monkeypatch.setattr(
        "auto_ontology.dal.schema_version.require_current_schema",
        lambda: events.append("schema"),
    )

    def bootstrap():
        events.append("bootstrap")
        return sources

    monkeypatch.setattr("auto_ontology.connectors.sources.bootstrap_sources", bootstrap)

    def connector(connection_id):
        assert events[:2] == ["schema", "bootstrap"]
        source = next(s for s in sources if s.connection_id == connection_id)
        events.append(("open", connection_id))
        result = Mock(
            connection_id=connection_id,
            database_name=source.database_name,
            dialect=source.dialect,
        )
        result.close.side_effect = lambda: events.append(("close", connection_id))
        return result

    monkeypatch.setattr("auto_ontology.connectors.registry.get_connector", connector)

    def write(connector):
        events.append(("ingest", connector.connection_id, connector.database_name))

    monkeypatch.setattr("auto_ontology.catalog.ingest_catalog", write)
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.create_connector",
        Mock(side_effect=AssertionError("raw URL constructors are forbidden")),
    )

    seed.ingest_catalog()

    assert events == [
        "schema",
        "bootstrap",
        ("open", "source-a"),
        ("ingest", "source-a", "pagila"),
        ("close", "source-a"),
        ("open", "source-b"),
        ("ingest", "source-b", "chinook"),
        ("close", "source-b"),
    ]


def test_all_six_sql_definitions_are_explicitly_owned_by_pagila(
    monkeypatch: pytest.MonkeyPatch, sources: list[SourceDescriptor]
) -> None:
    def create(**kwargs):
        assert kwargs["connection_id"] == "source-a", (
            "Chinook must never own these definitions"
        )
        return {"id": kwargs["name"]}

    attrs = Mock(side_effect=create)
    analyses = Mock(side_effect=create)
    monkeypatch.setattr(
        "auto_ontology.server.sql_attributes.service.create_sql_attribute", attrs
    )
    monkeypatch.setattr(
        "auto_ontology.server.custom_analyses.service.create_custom_analysis", analyses
    )
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.get_connector",
        Mock(side_effect=AssertionError("metadata selection cannot open connectors")),
    )

    seed.seed_sql_attributes({term: term for _, _, _, term, _ in seed.SQL_ATTRIBUTES})
    seed.seed_custom_analyses()

    assert attrs.call_count == 4
    assert analyses.call_count == 2


@pytest.mark.parametrize("seed_sql", ["attributes", "analyses"])
@pytest.mark.parametrize("pagila_count", [0, 2])
def test_missing_or_ambiguous_pagila_rejected_before_sql_writes(
    monkeypatch: pytest.MonkeyPatch,
    sources: list[SourceDescriptor],
    seed_sql: str,
    pagila_count: int,
) -> None:
    sources[:] = [sources[1]] + [
        SourceDescriptor(f"pagila-{i}", f"Films {i}", "pagila", "postgres", "env")
        for i in range(pagila_count)
    ]
    attrs = Mock(return_value={"id": "attribute"})
    analyses = Mock(return_value={"id": "analysis"})
    monkeypatch.setattr(
        "auto_ontology.server.sql_attributes.service.create_sql_attribute", attrs
    )
    monkeypatch.setattr(
        "auto_ontology.server.custom_analyses.service.create_custom_analysis", analyses
    )
    with pytest.raises(ValueError, match="pagila"):
        if seed_sql == "attributes":
            seed.seed_sql_attributes(
                {term: term for _, _, _, term, _ in seed.SQL_ATTRIBUTES}
            )
        else:
            seed.seed_custom_analyses()
    attrs.assert_not_called()
    analyses.assert_not_called()


@pytest.mark.parametrize("failure", ["schema", "bootstrap", "ingest"])
def test_startup_failure_is_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
    sources: list[SourceDescriptor],
    failure: str,
) -> None:
    error = RuntimeError(failure)
    check = Mock(side_effect=error if failure == "schema" else None)
    bootstrap = Mock(
        return_value=sources, side_effect=error if failure == "bootstrap" else None
    )
    connector = Mock(
        connection_id="source-a", database_name="pagila", dialect="postgres"
    )
    open_connector = Mock(return_value=connector)
    write = Mock(side_effect=error)
    monkeypatch.setattr(
        "auto_ontology.dal.schema_version.require_current_schema", check
    )
    monkeypatch.setattr("auto_ontology.connectors.sources.bootstrap_sources", bootstrap)
    monkeypatch.setattr(
        "auto_ontology.connectors.registry.get_connector", open_connector
    )
    monkeypatch.setattr("auto_ontology.catalog.ingest_catalog", write)

    with pytest.raises(RuntimeError, match=failure):
        seed.ingest_catalog()

    if failure == "schema":
        bootstrap.assert_not_called()
    if failure in ("schema", "bootstrap"):
        open_connector.assert_not_called()
    else:
        open_connector.assert_called_once_with("source-a")
        connector.close.assert_called_once_with()


def test_empty_env_does_not_fall_back_to_managed_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    bootstrap = Mock()
    monkeypatch.setattr("auto_ontology.connectors.sources.bootstrap_sources", bootstrap)
    with pytest.raises(SystemExit, match="CONNECTION_STRINGS is empty"):
        seed.ingest_catalog()
    bootstrap.assert_not_called()
