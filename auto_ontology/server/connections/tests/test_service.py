# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from auto_ontology.server.connections import service
from auto_ontology.server.connections.router import _serialize_connection


@pytest.fixture(autouse=True)
def managed_metadata(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Legacy adapters resolve metadata, not the list of every credential."""
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    rows = [
        {
            "id": f"id-{name}",
            "name": name,
            "display_label": name,
            "configuration_origin": "managed",
            "normalized_label": name,
        }
        for name in ("database", "main")
    ]
    with (
        patch.object(service, "list_source_metadata", return_value=rows),
        patch.object(
            service,
            "normalize_source_labels",
            side_effect=lambda values: [v.strip(" ").lower() for v in values],
        ),
        patch.object(
            service,
            "get_connection_storage",
            return_value={"connection": {}, "credential_ref": None},
        ),
    ):
        yield


def test_new_same_physical_source_uses_own_uuid_path() -> None:
    with (
        patch.object(
            service, "list_connections", return_value=[_snowflake_connection()]
        ),
        patch.object(service, "list_source_metadata", create=True, return_value=[]),
        patch.object(
            service, "normalize_source_labels", create=True, return_value=["database"]
        ),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        result = service.create_connection(connection=_snowflake_connection())
    assert write.call_args.args[0] == f"connections/{result['connection_id']}"
    assert insert.call_args.kwargs["credential_ref"] == write.call_args.args[0]
    assert result["database_name"] == "database"


def test_default_label_conflict_precedes_vault_write() -> None:
    with (
        patch.object(service, "list_connections", return_value=[]),
        patch.object(
            service,
            "list_source_metadata",
            create=True,
            return_value=[{"id": "other", "normalized_label": "database"}],
        ),
        patch.object(
            service, "normalize_source_labels", create=True, return_value=["database"]
        ),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(service, "insert_connection"),
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        with pytest.raises(ValueError, match="label_conflict"):
            service.create_connection(connection=_snowflake_connection())
    write.assert_not_called()


def test_label_only_does_not_touch_credentials_network_or_workers() -> None:
    with (
        patch.object(service, "update_connection_label") as rename,
        patch.object(service, "_stored_connection") as credentials,
        patch.object(service, "get_connection_storage") as storage,
        patch.object(service, "create_connector") as network,
        patch.object(service, "_refresh_connection_caches") as refresh,
        patch.object(service, "trigger_ingest") as ingest,
    ):
        result = service.update_connection(
            connection_id="id-database", display_label=" New label "
        )
    rename.assert_called_once_with(
        connection_id="id-database", display_label="New label"
    )
    for untouched in (credentials, storage, network, refresh, ingest):
        untouched.assert_not_called()
    assert result == {
        "connection_id": "id-database",
        "display_label": "New label",
        "database_name": "database",
        "connection": {},
    }


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_insert_compensates_only_new_uuid_path(
    cleanup_fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch.object(service, "_check_label", return_value="free"),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(
            service,
            "insert_connection",
            side_effect=RuntimeError("sensitive-driver-url"),
        ),
        patch.object(
            service,
            "delete_secrets",
            side_effect=RuntimeError("sensitive-vault-url") if cleanup_fails else None,
        ) as delete,
        patch.object(service, "trigger_ingest") as ingest,
    ):
        with pytest.raises(ValueError, match="source_conflict") as error:
            service.create_connection(_snowflake_connection())
    reference = write.call_args.args[0]
    assert reference.startswith("connections/")
    delete.assert_called_once_with(reference)
    ingest.assert_not_called()
    assert "sensitive" not in str(error.value) + caplog.text
    assert ("orphan_secret" in caplog.text) == cleanup_fails
    if cleanup_fails:
        assert reference.split("/")[1] in caplog.text


def test_failed_vault_write_does_not_activate_connection() -> None:
    with (
        patch.object(service, "_check_label", return_value="free"),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret", side_effect=RuntimeError("sensitive")),
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "delete_secrets") as delete,
    ):
        with pytest.raises(
            service.SourceUnavailableError, match="credential write failed"
        ) as error:
            service.create_connection(_snowflake_connection())
    assert "sensitive" not in str(error.value)
    insert.assert_not_called()
    delete.assert_not_called()


@pytest.mark.parametrize("operation", ["update", "sso", "delete"])
def test_existing_legacy_exact_ref_is_never_relocated(operation: str) -> None:
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ) as read,
        patch.object(
            service,
            "get_connection_storage",
            return_value={"connection": None, "credential_ref": "legacy/exact/sales"},
        ),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(service, "delete_secrets") as delete,
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "clear_connection") as clear,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as ingest,
        patch.object(service, "trigger_reset") as reset,
    ):
        if operation == "update":
            service.update_connection(
                connection_id="id-database",
                connection={**_snowflake_connection(), "password": ""},
            )
            assert write.call_args.args[1]["password"] == "secret"
        elif operation == "sso":
            service.set_sso_federation(connection_id="id-database", enabled=True)
            assert write.call_args.args[1]["sso_federation"] is True
        else:
            service.delete_connection(connection_id="id-database")
            delete.assert_called_once_with("legacy/exact/sales")
            clear.assert_called_once_with(connection_id="id-database")
            reset.assert_called_once_with("id-database")
        if operation != "delete":
            read.assert_called_once_with("id-database")
            assert write.call_args.args[0] == "legacy/exact/sales"
            assert insert.call_args.kwargs["credential_ref"] == "legacy/exact/sales"
            assert insert.call_args.kwargs["connection"] == ""
            delete.assert_not_called()
        ingest.assert_not_called()


def test_missing_selected_credentials_never_reaches_inline_write_or_network() -> None:
    with (
        patch.object(
            service, "load_source_connection", side_effect=ValueError("sensitive")
        ) as selected,
        patch.object(service, "list_connections") as all_credentials,
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "create_connector") as connector,
    ):
        with pytest.raises(service.SourceUnavailableError):
            service.test_connection(
                _snowflake_connection(), replacing_connection_id="id-database"
            )
    selected.assert_called_once_with("id-database")
    for untouched in (all_credentials, insert, connector):
        untouched.assert_not_called()


def test_legacy_name_ambiguity_is_metadata_only() -> None:
    rows = [
        {"id": identity, "name": "database", "configuration_origin": "managed"}
        for identity in ("one", "two")
    ]
    with (
        patch.object(service, "list_source_metadata", return_value=rows),
        patch.object(service, "_stored_connection") as read,
        patch.object(service, "clear_connection") as clear,
    ):
        with pytest.raises(service.AmbiguousConnectionError):
            service.delete_connection("database")
    read.assert_not_called()
    clear.assert_not_called()


@pytest.mark.parametrize("origin", ["env", None])
def test_nonmanaged_roots_cannot_be_mutated(origin: str | None) -> None:
    with patch.object(
        service,
        "list_source_metadata",
        return_value=[
            {"id": "id-database", "name": "database", "configuration_origin": origin}
        ],
    ):
        with pytest.raises(LookupError):
            service.update_connection(connection_id="id-database", display_label="new")


def test_env_mode_preserves_inactive_managed_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONNECTION_STRINGS", "synthetic.invalid")
    with patch.object(service, "insert_connection") as insert:
        with pytest.raises(ValueError, match="environment-managed"):
            service.create_connection(_snowflake_connection())
        with pytest.raises(ValueError, match="environment-managed"):
            service.delete_connection(connection_id="id-database")
    insert.assert_not_called()


@pytest.mark.parametrize("operation", ["update", "sso", "delete"])
def test_existing_reference_cannot_be_replaced_by_inline_when_vault_is_disabled(
    operation: str,
) -> None:
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(
            service,
            "get_connection_storage",
            return_value={"connection": None, "credential_ref": "legacy/exact"},
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "write_secret") as write,
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "clear_connection") as clear,
        patch.object(service, "delete_secrets") as delete,
    ):
        with pytest.raises(service.SourceUnavailableError):
            if operation == "update":
                service.update_connection(
                    connection_id="id-database", connection=_snowflake_connection()
                )
            elif operation == "sso":
                service.set_sso_federation(connection_id="id-database", enabled=True)
            else:
                service.delete_connection(connection_id="id-database")
    for untouched in (write, insert, clear, delete):
        untouched.assert_not_called()


@pytest.mark.parametrize("database", [None, "", "   "])
def test_create_requires_real_physical_name(database: str | None) -> None:
    with patch.object(service, "write_secret") as write:
        with pytest.raises(ValueError, match="Database name is required"):
            service.create_connection({**_snowflake_connection(), "database": database})
    write.assert_not_called()


def test_create_keeps_metadata_out_of_driver_json() -> None:
    connection = {
        **_snowflake_connection(),
        "connection_id": "untrusted",
        "display_label": "not-driver",
        "credential_ref": "other-secret",
        "origin": "env",
    }
    with (
        patch.object(service, "_check_label", return_value="free"),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        result = service.create_connection(connection)
    assert result["connection_id"] != "untrusted"
    assert json.loads(insert.call_args.kwargs["connection"]) == _snowflake_connection()


def test_sso_toggle_preserves_inline_storage_even_after_vault_is_enabled() -> None:
    stored = _snowflake_connection()
    with (
        patch.object(service, "_stored_connection", return_value=stored),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(service, "write_secret") as write,
        patch.object(service, "insert_connection") as insert,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as ingest,
    ):
        result = service.set_sso_federation(connection_id="id-database", enabled=True)
    assert json.loads(insert.call_args.kwargs["connection"]) == {
        **stored,
        "sso_federation": True,
    }
    assert insert.call_args.kwargs["credential_ref"] is None
    assert result["connection_id"] == "id-database"
    assert "sso_federation" not in stored
    write.assert_not_called()
    ingest.assert_not_called()


def _snowflake_connection() -> dict[str, object]:
    return {
        "type": "snowflake",
        "account": "account",
        "warehouse": "warehouse",
        "user": "user",
        "password": "secret",
        "database": "database",
        "schemas": ["GPU_FLEET"],
    }


def _kyuubi_connection() -> dict[str, object]:
    return {
        "type": "kyuubi",
        "host": "hive.example.com",
        "port": "10000",
        "user": "user",
        "database": "database",
        "truststore": "/etc/ssl/ca.jks",
        "truststore_password": "secret",
    }


def test_create_connection_persists_connection_without_vault() -> None:
    connection = _snowflake_connection()

    with (
        patch.object(service, "_check_label", return_value="database"),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(service, "refresh_chat_workers"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.create_connection(connection=connection)

    stored = json.loads(insert_connection.call_args.kwargs["connection"])
    assert stored["password"] == "secret"
    assert stored["schemas"] == ["GPU_FLEET"]
    trigger_ingest.assert_called_once_with(
        insert_connection.call_args.kwargs["connection_id"]
    )


def _serialize_driver_settings(settings: dict) -> dict:
    return _serialize_connection(
        {
            "connection_id": "source",
            "display_label": "Source",
            "database_name": settings["database"],
            "connection": settings,
        }
    )


def test_public_connection_payload_redacts_credentials() -> None:
    serialized = _serialize_driver_settings(
        {
            **_snowflake_connection(),
            "password_env": "SNOWFLAKE_PASSWORD",
        }
    )

    assert serialized["database_name"] == "database"
    assert "password" not in serialized["connection"]
    assert "password_env" not in serialized["connection"]


def test_public_connection_payload_redacts_the_private_key() -> None:
    serialized = _serialize_driver_settings(
        {
            **_snowflake_connection(),
            "private_key": "-----BEGIN PRIVATE KEY-----\nsecret\n",
            "private_key_passphrase": "phrase",
        }
    )

    assert "private_key" not in serialized["connection"]
    assert "private_key_passphrase" not in serialized["connection"]
    assert "BEGIN PRIVATE KEY" not in json.dumps(serialized)


def test_public_connection_payload_redacts_the_truststore_password() -> None:
    """Kyuubi's truststore password unlocks the keystore, so it is a credential
    like any other -- only the truststore path itself is safe to show."""
    serialized = _serialize_driver_settings(_kyuubi_connection())

    assert "truststore_password" not in serialized["connection"]
    assert serialized["connection"]["truststore"] == "/etc/ssl/ca.jks"


class _StubConnector:
    def __init__(self, schemas: list[str]) -> None:
        self._schemas = schemas
        self.closed = False

    def get_schemas(self) -> list[str]:
        return self._schemas

    def close(self) -> None:
        self.closed = True


def _databricks_connection(**extra: object) -> dict[str, object]:
    return {
        "type": "databricks",
        "host": "example.databricks.com",
        "http_path": "/sql/1.0/warehouses/w",
        "password": "secret",
        "database": "main",
        **extra,
    }


def test_test_connection_accepts_a_schema_that_exists() -> None:
    connector = _StubConnector(["Sales", "marketing"])

    with (
        patch.object(service, "_check_label", return_value="database"),
        patch.object(service, "create_connector", return_value=connector),
    ):
        # Case-insensitive: Databricks reports its own casing.
        schemas = service.test_connection(_databricks_connection(schema="sales"))

    assert schemas == ["Sales", "marketing"]
    assert connector.closed


def test_test_connection_rejects_a_schema_that_does_not_exist() -> None:
    """The UI skips schema selection once a schema is named, so a typo has to fail
    here — otherwise it surfaces later as an ingest that silently finds nothing."""
    connector = _StubConnector(["sales"])

    with (
        patch.object(service, "_check_label", return_value="database"),
        patch.object(service, "create_connector", return_value=connector),
    ):
        try:
            service.test_connection(_databricks_connection(schema="typo_schema"))
        except ValueError as exc:
            assert "typo_schema" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the missing schema")

    assert connector.closed


def test_test_connection_allows_the_connection_an_edit_replaces() -> None:
    """Without `replacing`, re-testing an existing connection fails the duplicate
    check — an edit could never be validated before saving."""
    connector = _StubConnector(["sales"])
    stored = _databricks_connection()

    with (
        patch.object(service, "_stored_connection", return_value=stored),
        patch.object(service, "create_connector", return_value=connector),
    ):
        schemas = service.test_connection(
            _databricks_connection(password=""), replacing="main"
        )

    assert schemas == ["sales"]


def test_test_connection_rejects_renaming_the_database_it_replaces() -> None:
    with patch.object(
        service, "_stored_connection", return_value=_databricks_connection()
    ):
        try:
            service.test_connection(
                _databricks_connection(database="other"), replacing="main"
            )
        except ValueError as exc:
            assert "cannot be changed" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the renamed database")


def test_update_connection_keeps_the_stored_secret_when_left_blank() -> None:
    """The UI never receives credentials, so it cannot send them back; a blank
    field means "unchanged" rather than "clear it"."""
    stored = _snowflake_connection()

    with (
        patch.object(service, "_stored_connection", return_value=stored),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "password": "", "user": "renamed"},
        )

    updated = json.loads(insert_connection.call_args.kwargs["connection"])
    assert updated["password"] == "secret"
    assert updated["user"] == "renamed"
    # The schema allowlist is untouched, so the ingested graph is still correct.
    trigger_ingest.assert_not_called()


def test_update_connection_keeps_the_stored_truststore_password() -> None:
    """Redacted on the way out like every other credential, so the edit form has
    none to send back and a blank field must not wipe it."""
    with (
        patch.object(service, "_stored_connection", return_value=_kyuubi_connection()),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest"),
    ):
        service.update_connection(
            database_name="database",
            connection={
                **_kyuubi_connection(),
                "truststore_password": "",
                "host": "hive2.example.com",
            },
        )

    updated = json.loads(insert_connection.call_args.kwargs["connection"])
    assert updated["truststore_password"] == "secret"
    assert updated["host"] == "hive2.example.com"


def test_update_connection_reingests_even_when_a_cache_refresh_fails() -> None:
    """Caches are refreshed after the write lands, so a failure there must not
    swallow the ingest: the new allowlist is already stored, and a retry would
    compare it against itself and skip the ingest for good."""
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection"),
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(
            service, "refresh_chat_workers", side_effect=RuntimeError("no workers")
        ),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "schemas": ["GPU_FLEET", "BILLING"]},
        )

    trigger_ingest.assert_called_once()


def test_update_connection_reingests_when_the_schema_allowlist_changes() -> None:
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection"),
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        service.update_connection(
            database_name="database",
            connection={**_snowflake_connection(), "schemas": ["GPU_FLEET", "BILLING"]},
        )

    trigger_ingest.assert_called_once_with("id-database")


def test_update_connection_rejects_changing_the_connector_type() -> None:
    """The ingested graph was built by the stored connector, so pointing the same
    catalog row at a different driver is a new connection, not an edit."""
    with patch.object(
        service, "_stored_connection", return_value=_snowflake_connection()
    ):
        try:
            service.update_connection(
                database_name="database",
                connection={**_snowflake_connection(), "type": "postgresql"},
            )
        except ValueError as exc:
            assert "cannot be changed" in str(exc)
        else:
            raise AssertionError("expected a ValueError for the changed type")


def test_update_connection_reports_an_unknown_database_as_missing() -> None:
    with patch.object(service, "_stored_connection", return_value=None):
        try:
            service.update_connection(
                database_name="ghost", connection=_snowflake_connection()
            )
        except LookupError as exc:
            assert "source_not_found" in str(exc)
        else:
            raise AssertionError("expected a LookupError for the unknown database")


def test_delete_connection_clears_the_stored_record() -> None:
    """The record has to go from this process. Leaving it to the ingestion
    service meant a delete did nothing whenever that service was unreachable,
    and the connection came back on the next page load."""
    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "clear_connection") as clear_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_reset") as trigger_reset,
    ):
        result = service.delete_connection("database")

    clear_connection.assert_called_once_with(connection_id="id-database")
    trigger_reset.assert_called_once_with("id-database")
    assert result == {
        "connection_id": "id-database",
        "display_label": "database",
        "database_name": "database",
    }


def test_delete_connection_removes_the_vault_secret_before_the_row() -> None:
    """`list_connections` prefers a Vault secret over the row, so clearing the
    row first would resurrect the connection if the secret delete then failed."""
    calls: list[str] = []

    with (
        patch.object(
            service, "_stored_connection", return_value=_snowflake_connection()
        ),
        patch.object(service, "is_vault_configured", return_value=True),
        patch.object(
            service,
            "get_connection_storage",
            return_value={"connection": None, "credential_ref": "legacy/database"},
        ),
        patch.object(
            service, "delete_secrets", side_effect=lambda *_: calls.append("vault")
        ),
        patch.object(
            service, "clear_connection", side_effect=lambda **_: calls.append("row")
        ),
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_reset"),
    ):
        service.delete_connection("database")

    assert calls == ["vault", "row"]


def test_delete_connection_reports_an_unknown_database_as_missing() -> None:
    with (
        patch.object(service, "_stored_connection", return_value=None),
        patch.object(service, "clear_connection") as clear_connection,
        patch.object(service, "trigger_reset") as trigger_reset,
    ):
        assert service.delete_connection("ghost") is None

    clear_connection.assert_not_called()
    trigger_reset.assert_not_called()


def test_create_connection_drops_the_schema_shorthand() -> None:
    """`schema` is the form's shorthand for `schemas`; storing both would leave two
    sources of truth for what gets ingested."""
    connection = _databricks_connection(schema="sales", schemas=["sales"])

    with (
        patch.object(service, "_check_label", return_value="main"),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "invalidate_connectors_cache"),
        patch.object(service, "refresh_chat_workers"),
        patch.object(service, "trigger_ingest"),
    ):
        service.create_connection(connection=connection)

    stored = json.loads(insert_connection.call_args.kwargs["connection"])
    assert "schema" not in stored
    assert stored["schemas"] == ["sales"]


def test_same_name_fixture_preserves_legacy_ids_and_separates_samples(
    same_name_sources: dict,
) -> None:
    from uuid import UUID

    local, remote = same_name_sources["sources"]
    legacy = same_name_sources["legacy_snapshot"]
    (root,) = legacy["catalog_database"]
    (schema,) = legacy["catalog_schema"]
    (table,) = legacy["catalog_table"]
    assert (
        root["id"] == local["connection_id"] == "11111111-1111-4111-8111-111111111111"
    )
    assert schema["id"] == "11111111-1111-4111-8111-111111111112"
    assert schema["database_id"] == root["id"]
    assert table["id"] == local["table"]["id"] == "11111111-1111-4111-8111-111111111113"
    assert table["schema_id"] == schema["id"]
    assert [column["id"] for column in legacy["catalog_column"]] == [
        "11111111-1111-4111-8111-111111111114",
        "11111111-1111-4111-8111-111111111115",
        "11111111-1111-4111-8111-111111111116",
    ]
    for column in legacy["catalog_column"]:
        assert column["table_id"] == table["id"]
        assert json.loads(column["sample_values"]) == [
            local["table"]["sample_rows"][0][column["name"]]
        ]
    assert UUID(local["connection_id"]) != UUID(remote["connection_id"])
    assert local["connection_label"] != remote["connection_label"]
    assert local["connection"]["host"] != remote["connection"]["host"]
    assert (
        local["connection"]["database"] == remote["connection"]["database"] == "sales"
    )
    assert local["table"]["name"] == remote["table"]["name"] == "orders"
    assert local["table"]["schema_name"] == remote["table"]["schema_name"] == "public"
    assert "draft_status" in local["table"]["columns"]
    assert "draft_status" not in json.dumps(remote)
    assert local["table"]["sample_rows"][0]["amount"] == 111
    assert remote["table"]["sample_rows"][0]["amount"] == 222
    assert "LOCAL_DRAFT_ONLY" not in json.dumps(remote)
    assert "REMOTE_SHIPPED_ONLY" not in json.dumps(local)
    assert set(same_name_sources["shared_term"]["table_ids"]) == {
        local["table"]["id"],
        remote["table"]["id"],
    }
    variants = legacy["credential_storage_variants"]
    assert variants["inline"]["root_id"] == variants["vault"]["root_id"] == root["id"]
    assert variants["inline"]["connection_ref"] == "sources[0].connection"
    assert variants["vault"]["connection"] is None
    assert variants["vault"]["credential_ref"] == "test-dummy/connections/sales"
    assert all(chat["connection_id"] is None for chat in legacy["conversations"])


def test_create_distinct_source_with_same_physical_database(
    same_name_sources: dict,
) -> None:
    from copy import deepcopy
    from uuid import UUID

    from auto_ontology.server.connections import router

    local, remote = same_name_sources["sources"]
    existing = deepcopy(local["connection"])
    before = deepcopy(existing)
    legacy_before = deepcopy(same_name_sources["legacy_snapshot"])
    body = router.ConnectionBody.model_validate(
        {
            "connection": deepcopy(remote["connection"]),
            "display_label": remote["connection_label"],
        }
    )

    with (
        patch.object(service, "list_connections", return_value=[existing]),
        patch.object(service, "is_vault_configured", return_value=False),
        patch.object(service, "write_secret") as write_secret,
        patch.object(service, "insert_connection") as insert_connection,
        patch.object(service, "_refresh_connection_caches"),
        patch.object(service, "trigger_ingest") as trigger_ingest,
    ):
        try:
            response = router.create_connection(body)
        finally:
            assert existing == before
            assert same_name_sources["legacy_snapshot"] == legacy_before
            write_secret.assert_not_called()

    created = response["data"]
    assert UUID(created["connection_id"]) != UUID(local["connection_id"])
    assert created["display_label"] == remote["connection_label"]
    assert created["database_name"] == "sales"
    assert created["connection"] == {
        key: value
        for key, value in remote["connection"].items()
        if key not in service.SECRET_FIELDS
    }
    insert_connection.assert_called_once()
    persisted = insert_connection.call_args.kwargs
    assert persisted["display_label"] == remote["connection_label"]
    assert persisted["connection_id"] == created["connection_id"]
    assert json.loads(persisted["connection"]) == remote["connection"]
    trigger_ingest.assert_called_once()
