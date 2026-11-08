# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling must never resolve or blend same-name source identities."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.semantic import visit_enter as ve
from auto_ontology.connectors.sources import SourceDescriptor


def source(cid: str) -> SourceDescriptor:
    return SourceDescriptor(cid, cid, "sales", "postgres", "managed")


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    monkeypatch.setattr(
        ve,
        "get_connectors",
        Mock(side_effect=AssertionError("global connector scan forbidden")),
        raising=False,
    )
    monkeypatch.setattr(
        ve, "list_active_sources", lambda: [source("A"), source("B")], raising=False
    )
    spies = {}
    for name in (
        "get_connector",
        "suggest_potential_foreign_keys",
        "extract_term",
        "store_column_sample_values",
        "upsert_table_term",
    ):
        spies[name] = Mock(side_effect=AssertionError(name))
        monkeypatch.setattr(ve, name, spies[name], raising=False)
    return spies


def test_resolver_opens_only_selected_source(boundary: dict[str, Mock]) -> None:
    connector = SimpleNamespace(connection_id="A", database_name="sales")
    boundary["get_connector"].side_effect = lambda cid: (
        connector if cid == "A" else (_ for _ in ()).throw(AssertionError("B opened"))
    )
    assert ve._resolve_connector("A") is connector
    boundary["get_connector"].assert_called_once_with("A")


def test_breaker_is_exact_source_scoped() -> None:
    breaker = ve._SamplingCircuitBreaker()
    a = SimpleNamespace(connection_id="A", database_name="sales")
    b = SimpleNamespace(connection_id="a", database_name="sales")
    for _ in range(ve._MAX_CONSECUTIVE_SAMPLE_FAILURES):
        breaker.record_failure(b, TimeoutError())
    assert not breaker.is_open(a)
    breaker.reset("A")
    assert breaker.is_open(b)
    breaker.reset("a")
    assert not breaker.is_open(b)


@pytest.mark.parametrize(
    "table_cid,ctx_cid,cid",
    [
        ("B", "A", "A"),
        ("A", "B", "A"),
        (None, "A", "A"),
        ("A", None, "A"),
        ("X", "X", "X"),
        ("", "", ""),
    ],
)
def test_process_rejects_unproven_source_before_side_effects(
    boundary: dict[str, Mock], table_cid: str | None, ctx_cid: str | None, cid: str
) -> None:
    with pytest.raises(ValueError, match="source_"):
        ve.process_table(
            {"id": "t", "name": "orders", "connection_id": table_cid},
            {"connection_id": ctx_cid},
            connection_id=cid,
            domain_summary=None,
        )
    for spy in boundary.values():
        spy.assert_not_called()


@pytest.mark.parametrize("connector_cid", [None, "B"])
def test_profiling_rejects_connector_before_sql_store(
    boundary: dict[str, Mock], connector_cid: str | None
) -> None:
    connector = Mock(connection_id=connector_cid)
    with pytest.raises(ValueError, match="source_"):
        ve.calculate_columns_profiling(
            {"id": "t", "name": "orders", "connection_id": "A"}, [], connector
        )
    connector.execute.assert_not_called()
    boundary["store_column_sample_values"].assert_not_called()


def test_wrong_embedder_rejected_before_model_or_connector(
    boundary: dict[str, Mock],
) -> None:
    with pytest.raises(ValueError, match="source_"):
        ve.process_table(
            {"id": "t", "name": "orders", "connection_id": "A"},
            {"connection_id": "A"},
            connection_id="A",
            domain_summary=None,
            embedder=SimpleNamespace(connection_id="B"),
        )
    for spy in boundary.values():
        spy.assert_not_called()


def test_selected_connector_failure_never_scans_neighbours(
    boundary: dict[str, Mock],
) -> None:
    boundary["get_connector"].side_effect = RuntimeError("selected source unavailable")
    with pytest.raises(RuntimeError, match="selected source unavailable"):
        ve.process_table(
            {"id": "t", "name": "orders", "connection_id": "A"},
            {"connection_id": "A"},
            connection_id="A",
            domain_summary=None,
        )
    boundary["get_connector"].assert_called_once_with("A")
    boundary["extract_term"].assert_not_called()


def test_connector_mismatch_rejected_before_model_sql_store(
    boundary: dict[str, Mock],
) -> None:
    connector = Mock(connection_id="B", database_name="sales")
    boundary["get_connector"].side_effect = None
    boundary["get_connector"].return_value = connector
    with pytest.raises(ValueError, match="source_conflict"):
        ve.process_table(
            {"id": "t", "name": "orders", "connection_id": "A"},
            {"connection_id": "A"},
            connection_id="A",
            domain_summary=None,
        )
    connector.execute.assert_not_called()
    boundary["extract_term"].assert_not_called()
    boundary["store_column_sample_values"].assert_not_called()


def test_probe_without_connector_identity_never_executes() -> None:
    connector = Mock(connection_id=None)
    with pytest.raises(ValueError, match="source_not_ready"):
        ve._execute_capped(connector, "SELECT 1")
    connector.execute.assert_not_called()
