# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.connectors import databricks_oauth as oauth
from auto_ontology.connectors import registry, sources
from auto_ontology.dal import connections


@pytest.fixture
def setup_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple]:
    active = [
        sources.SourceDescriptor(i, i, "sales", "databricks", "managed")
        for i in ("a", "b")
    ]
    settings = {
        i: {
            "type": "databricks",
            "host": i,
            "database": "sales",
            "password": "stored",
            "http_path": "/sql",
            "schemas": [" public ", ""],
            "table_deny_regex": "^tmp_",
        }
        for i in ("a", "b")
    }
    reader = Mock(side_effect=lambda i: settings[i])
    factory = Mock(
        side_effect=lambda url, **kw: SimpleNamespace(
            database_name="sales", url=url, close=Mock(), **kw
        )
    )
    exchange = Mock(side_effect=lambda host, token: f"{host}-{token}")
    monkeypatch.setattr(sources, "list_active_sources", lambda: active)
    monkeypatch.setattr(connections, "load_source_connection", reader)
    monkeypatch.setattr(
        connections,
        "list_connections",
        Mock(side_effect=AssertionError("unscoped credentials")),
    )
    monkeypatch.setattr(registry, "create_connector", factory)
    monkeypatch.setattr(oauth, "exchange_subject_token", exchange)
    registry.invalidate_connectors_cache()
    yield active, settings, reader, factory, exchange
    registry.invalidate_connectors_cache()


def test_same_physical_name_has_separate_id_cache_and_closes(
    setup_registry: tuple,
) -> None:
    active, _, reader, _, _ = setup_registry
    a = registry.get_connector("a")
    b = registry.get_connector("b")
    assert a is not b
    assert (a.connection_id, b.connection_id) == ("a", "b")
    assert a.database_name == b.database_name == "sales"
    assert registry.get_connector("a") is a
    assert registry.get_connectors() == [a, b]
    assert reader.call_count == 2
    assert a.schemas == ["public"]
    assert not a.table_filter.keep("tmp_orders")
    active.pop(0)
    with pytest.raises(ValueError, match="source_unavailable"):
        registry.get_connector("a")
    with pytest.raises(ValueError, match="source_unavailable"):
        registry.get_connectors_for_subject_token("caller", "a")
    assert reader.call_count == 2
    registry.invalidate_connectors_cache()
    a.close.assert_called_once()
    b.close.assert_called_once()
    assert registry.get_connector("b") is not b


def test_selected_nonfederated_never_touches_b_or_exchange(
    setup_registry: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, settings, reader, _, exchange = setup_registry
    settings["b"]["sso_federation"] = True
    reader.side_effect = lambda i: (
        settings["a"] if i == "a" else pytest.fail("B touched")
    )
    federation_check = Mock(
        side_effect=lambda conn: (
            False if conn["host"] == "a" else pytest.fail("B SSO touched")
        )
    )
    monkeypatch.setattr(oauth, "uses_sso_federation", federation_check)
    a = registry.get_connectors_for_subject_token(None, "a")[0]
    assert registry.get_connectors_for_subject_token("caller", "a") == [a]
    assert reader.call_args_list == [(("a",),)]
    exchange.assert_not_called()
    assert all(call.args[0]["host"] == "a" for call in federation_check.call_args_list)


def test_selected_federation_is_fresh_per_caller_and_not_service_pat(
    setup_registry: tuple,
) -> None:
    _, settings, reader, factory, exchange = setup_registry
    settings["a"]["sso_federation"] = True
    stored = registry.get_connector("a")
    a = registry.get_connectors_for_subject_token("alice", "a")[0]
    b = registry.get_connectors_for_subject_token("bob", "a")[0]
    assert a is not b and a is not stored
    assert "a-alice" in a.url and "a-bob" in b.url
    assert a.connection_id == b.connection_id == "a"
    assert a.database_name == b.database_name == "sales"
    assert a.schemas == ["public"] and not a.table_filter.keep("tmp_orders")
    assert reader.call_count == 1
    assert exchange.call_count == 2 and factory.call_count == 3


@pytest.mark.parametrize("token", [None, "rejected"])
def test_selected_federation_fails_closed(
    setup_registry: tuple, token: str | None
) -> None:
    _, settings, _, factory, exchange = setup_registry
    settings["b"]["sso_federation"] = True
    exchange.side_effect = oauth.DatabricksOAuthError("rejected")
    with pytest.raises(oauth.DatabricksOAuthError):
        registry.get_connectors_for_subject_token(token, "b")
    factory.assert_not_called()


def test_no_selector_rejects_multi_before_credentials(setup_registry: tuple) -> None:
    _, _, reader, factory, exchange = setup_registry
    with pytest.raises(ValueError, match="source_required"):
        registry.get_connectors_for_subject_token("caller")
    reader.assert_not_called()
    factory.assert_not_called()
    exchange.assert_not_called()


def test_single_source_legacy_and_empty_source(setup_registry: tuple) -> None:
    active, _, reader, _, _ = setup_registry
    active.pop()
    assert registry.get_connectors_for_subject_token(None)[0].connection_id == "a"
    active.clear()
    with pytest.raises(ValueError, match="source_unavailable"):
        registry.get_connectors_for_subject_token(None)
    assert reader.call_count == 1


def test_selected_failure_does_not_fallback_but_explicit_enumeration_skips(
    setup_registry: tuple,
) -> None:
    _, settings, reader, factory, _ = setup_registry
    reader.side_effect = lambda i: (
        settings[i] if i == "a" else (_ for _ in ()).throw(RuntimeError("secret-url"))
    )
    with pytest.raises(ValueError, match="source_unavailable") as error:
        registry.get_connector("b")
    assert "secret-url" not in str(error.value)
    factory.assert_not_called()
    assert [c.connection_id for c in registry.get_connectors()] == ["a"]


@pytest.mark.parametrize("explicit", [False, True])
def test_env_selected_url_strips_metadata_without_managed_read(
    setup_registry: tuple, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    active, _, reader, _, exchange = setup_registry
    source_id = "10000000-0000-4000-8000-000000000001"
    active[:] = [
        sources.SourceDescriptor(source_id, "Production", "sales", "postgres", "env")
    ]
    selectors = (
        f"?metadata_connection_id={source_id}&metadata_label=Production&sslmode=require"
        if explicit
        else "?sslmode=require"
    )
    monkeypatch.setenv(
        "CONNECTION_STRINGS", "postgresql://user:secret@localhost/sales" + selectors
    )
    connector = registry.get_connectors_for_subject_token(None, source_id)[0]
    assert connector.url == "postgresql://user:secret@localhost/sales?sslmode=require"
    assert connector.connection_id == source_id
    reader.assert_not_called()
    exchange.assert_not_called()


def test_factory_errors_do_not_expose_driver_secrets(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(
        registry.CONNECTOR_REGISTRY,
        "broken",
        Mock(side_effect=RuntimeError("password=secret")),
    )
    with pytest.raises(ValueError) as error:
        registry.create_connector("broken://user:secret@host/sales")
    assert "secret" not in str(error.value)
    assert "secret" not in caplog.text


@pytest.mark.parametrize("result", ["", RuntimeError("secret")])
def test_failed_or_empty_exchange_never_uses_stored_pat(
    setup_registry: tuple, result: str | Exception
) -> None:
    _, settings, reader, factory, exchange = setup_registry
    settings["a"]["sso_federation"] = True
    if isinstance(result, Exception):
        exchange.side_effect = result
    else:
        exchange.side_effect = None
        exchange.return_value = result
    with pytest.raises(oauth.DatabricksOAuthError) as error:
        registry.get_connectors_for_subject_token("caller", "a")
    assert "secret" not in str(error.value)
    factory.assert_not_called()
    reader.assert_called_once_with("a")


def test_invalidation_closes_other_source_even_if_one_close_fails(
    setup_registry: tuple,
) -> None:
    a = registry.get_connector("a")
    b = registry.get_connector("b")
    a.close.side_effect = RuntimeError("secret")
    registry.invalidate_connectors_cache()
    b.close.assert_called_once()
    assert not registry._connectors and not registry._connection_specs


def test_same_name_env_selection_uses_id_not_url_order(
    setup_registry: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    active, _, reader, factory, _ = setup_registry
    a, b = (
        "10000000-0000-4000-8000-000000000001",
        "10000000-0000-4000-8000-000000000002",
    )
    active[:] = [
        sources.SourceDescriptor(i, i, "sales", "postgres", "env") for i in (a, b)
    ]
    monkeypatch.setenv(
        "CONNECTION_STRINGS",
        "\n".join(
            f"postgresql://user:secret@{host}/sales?metadata_connection_id={i}&metadata_label={host}"
            for i, host in ((b, "b"), (a, "a"))
        ),
    )
    connector = registry.get_connector(a)
    assert connector.url == "postgresql://user:secret@a/sales"
    assert connector.connection_id == a and connector.database_name == "sales"
    factory.assert_called_once()
    reader.assert_not_called()
