# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy.dialects import postgresql

from auto_ontology.dal import connections
from auto_ontology.dal.tests.test_connection_identity_schema import identity_database  # noqa: F401


@pytest.mark.parametrize("encoded", [False, True])
def test_selected_inline_reads_only_id_and_managed_origin(
    monkeypatch: pytest.MonkeyPatch, encoded: bool
) -> None:
    payload = {
        "type": "postgres",
        "database": "sales",
        "password": "own",
        "display_label": "Prod",
        "connection_id": "a",
        "database_name": "not-driver",
        "credential_ref": "not-driver",
    }
    store = Mock()
    store.query_read.return_value = [
        {
            "connection": json.dumps(payload) if encoded else payload,
            "credential_ref": None,
        }
    ]
    secret = Mock(side_effect=AssertionError("no name fallback"))
    monkeypatch.setattr(connections, "store", lambda: store)
    monkeypatch.setattr(connections, "read_secret", secret)
    result = connections.load_source_connection("a")
    assert result == {"type": "postgres", "database": "sales", "password": "own"}
    statement = store.query_read.call_args.args[0].compile(dialect=postgresql.dialect())
    assert set(statement.params.values()) == {"a", "managed"}
    sql = str(statement)
    assert "catalog_database.id =" in sql and "configuration_origin =" in sql
    assert "catalog_database.name" not in sql
    secret.assert_not_called()


@pytest.mark.parametrize(
    "value", [None, "", [], {}, "[]", "null", "bad-json", {"password": "secret"}]
)
def test_invalid_selected_inline_is_safe_unavailable(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    store = Mock()
    store.query_read.return_value = [{"connection": value, "credential_ref": None}]
    monkeypatch.setattr(connections, "store", lambda: store)
    monkeypatch.setattr(
        connections, "read_secret", Mock(side_effect=AssertionError("fallback"))
    )
    with pytest.raises(ValueError, match="source_unavailable") as error:
        connections.load_source_connection("a")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "secret_value",
    [{"type": "postgres", "password": "vault"}, "", RuntimeError("secret")],
)
def test_exact_credential_ref_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, secret_value: Any
) -> None:
    store = Mock()
    store.query_read.return_value = [
        {
            "connection": {"type": "postgres", "password": "inline"},
            "credential_ref": "legacy/exact-path",
        }
    ]
    secret = (
        Mock(side_effect=secret_value)
        if isinstance(secret_value, Exception)
        else Mock(return_value=secret_value)
    )
    monkeypatch.setattr(connections, "store", lambda: store)
    monkeypatch.setattr(connections, "read_secret", secret)
    if isinstance(secret_value, dict):
        assert connections.load_source_connection("a") == secret_value
    else:
        with pytest.raises(ValueError, match="source_unavailable"):
            connections.load_source_connection("a")
    secret.assert_called_once_with("legacy/exact-path")


def test_unknown_selected_id_does_not_read_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Mock()
    store.query_read.return_value = []
    secret = Mock()
    monkeypatch.setattr(connections, "store", lambda: store)
    monkeypatch.setattr(connections, "read_secret", secret)
    with pytest.raises(ValueError, match="source_unavailable"):
        connections.load_source_connection("missing")
    secret.assert_not_called()


def test_real_store_same_name_rows_resolve_only_selected_credentials(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alembic import command
    from alembic.config import Config

    from auto_ontology.dal import schema
    from auto_ontology.dal.session import dispose_engine, store
    from auto_ontology.dal.tests.test_connection_identity_schema import ROOT

    request.getfixturevalue("identity_database")
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        for source_id, origin, ref in [
            ("a", "managed", None),
            ("b", "managed", "legacy/exact"),
            ("env", "env", None),
            ("catalog", None, None),
        ]:
            store().query_write(
                schema.catalog_database.insert().values(
                    id=source_id,
                    name="sales",
                    display_label=source_id,
                    configuration_origin=origin,
                    credential_ref=ref,
                    connection={
                        "type": "postgres",
                        "database": "sales",
                        "password": source_id,
                    },
                )
            )
        secret = Mock(
            return_value={
                "type": "postgres",
                "database": "sales",
                "password": "vault-b",
            }
        )
        monkeypatch.setattr(connections, "read_secret", secret)
        assert connections.load_source_connection("a")["password"] == "a"
        secret.assert_not_called()
        assert connections.load_source_connection("b")["password"] == "vault-b"
        secret.assert_called_once_with("legacy/exact")
        secret.reset_mock()
        for source_id in ("env", "catalog", "unknown"):
            with pytest.raises(ValueError, match="source_unavailable"):
                connections.load_source_connection(source_id)
        secret.assert_not_called()
        secret.return_value = ""
        with pytest.raises(ValueError, match="source_unavailable"):
            connections.load_source_connection("b")
    finally:
        dispose_engine()
