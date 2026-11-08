# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine

from auto_ontology.dal.tests.test_connection_identity_schema import (
    ADDITIVE,
    ROOT,
    identity_database,  # noqa: F401 -- shared opt-in disposable database fixture
)


@pytest.mark.parametrize(
    "mapping",
    [
        {"source": {"password": "MUST_NOT_BE_ACCEPTED"}},
        {"source": {"configuration_origin": "other"}},
        {"source": {"display_label": "   "}},
    ],
)
def test_operator_mapping_accepts_only_identity_metadata(mapping: dict) -> None:
    from dev_tools.migrate_connection_identity import validate_mapping

    with pytest.raises(ValueError):
        validate_mapping(mapping)


def _legacy_store() -> Engine:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text

    from auto_ontology.dal.session import sqlalchemy_url

    command.upgrade(Config(str(ROOT / "alembic.ini")), ADDITIVE)
    engine = create_engine(sqlalchemy_url())
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO catalog_database (id, name, connection) "
                "VALUES ('local', 'sales', CAST(:connection AS jsonb))"
            ),
            {"connection": json.dumps({"password": "TEST_DUMMY_SECRET"})},
        )
        conn.execute(
            text(
                "INSERT INTO catalog_schema (id, database_id, name) "
                "VALUES ('schema', 'local', 'public')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO catalog_table (id, schema_id, name, description) "
                "VALUES ('table', 'schema', 'orders', 'curated orders')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sql_query (id, sql_full_query) "
                "VALUES ('owned-query', 'SELECT * FROM orders'), "
                "('unknown-query', 'SELECT 42')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sql_query__table (sql_query_id, table_id) "
                "VALUES ('owned-query', 'table')"
            )
        )
        conn.execute(
            text("INSERT INTO sql_attribute (id, name) VALUES ('attribute', 'revenue')")
        )
        conn.execute(
            text(
                "INSERT INTO sql_attribute__sql (attribute_id, sql_query_id) "
                "VALUES ('attribute', 'owned-query')"
            )
        )
    return engine


def _run(*args: str) -> tuple[int, dict]:
    result = subprocess.run(
        [sys.executable, "-m", "dev_tools.migrate_connection_identity", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert "TEST_DUMMY_SECRET" not in result.stdout + result.stderr
    assert result.stdout, result.stderr
    return result.returncode, json.loads(result.stdout)


@pytest.mark.usefixtures("identity_database")
def test_check_is_read_only_and_apply_is_idempotent() -> None:
    from sqlalchemy import text

    engine = _legacy_store()
    try:
        status, report = _run("--check")
        assert status == 0
        assert report["pending_updates"] == 3
        assert report["unresolved_sql"]["sql_query"] == ["unknown-query"]
        assert report["ready_for_querying"] is False
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text("SELECT display_label FROM catalog_database")
                ).scalar_one()
                is None
            )
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM sql_query WHERE connection_id IS NOT NULL"
                    )
                ).scalar_one()
                == 0
            )
        status, applied = _run("--apply")
        assert status == 0 and applied["applied"] is True
        status, repeated = _run("--apply")
        assert status == 0 and repeated["pending_updates"] == 0
        assert repeated["ready_for_querying"] is False
        with engine.connect() as conn:
            assert conn.execute(
                text(
                    "SELECT id, name, display_label, configuration_origin FROM catalog_database"
                )
            ).one() == ("local", "sales", "sales", "managed")
            assert conn.execute(
                text("SELECT id, description FROM catalog_table")
            ).one() == ("table", "curated orders")
            assert (
                conn.execute(
                    text("SELECT connection_id FROM sql_attribute")
                ).scalar_one()
                == "local"
            )
            assert (
                conn.execute(
                    text(
                        "SELECT connection_id FROM sql_query WHERE id = 'unknown-query'"
                    )
                ).scalar_one()
                is None
            )
    finally:
        engine.dispose()


@pytest.mark.usefixtures("identity_database")
def test_label_conflict_does_not_partially_apply(tmp_path: Path) -> None:
    from sqlalchemy import text

    engine = _legacy_store()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO catalog_database (id, name, connection) VALUES ('other', 'Sales', '{}'::jsonb)"
                )
            )
        status, report = _run("--apply")
        assert status == 2
        assert {issue["code"] for issue in report["issues"]} == {"label_conflict"}
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM catalog_database WHERE display_label IS NOT NULL"
                    )
                ).scalar_one()
                == 0
            )
        mapping = tmp_path / "mapping.json"
        mapping.write_text(json.dumps({"other": {"display_label": "Other sales"}}))
        status, report = _run("--apply", "--mapping", str(mapping))
        assert status == 0 and report["applied"] is True
    finally:
        engine.dispose()


@pytest.mark.usefixtures("identity_database")
def test_ambiguous_origin_requires_explicit_legacy_reference(tmp_path: Path) -> None:
    from sqlalchemy import text

    engine = _legacy_store()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO catalog_database (id, name) VALUES ('vault', 'legacy')"
                )
            )
        status, report = _run("--apply")
        assert status == 2
        assert {issue["code"] for issue in report["issues"]} == {"origin_required"}
        mapping = tmp_path / "mapping.json"
        mapping.write_text(
            json.dumps(
                {
                    "vault": {
                        "configuration_origin": "managed",
                        "credential_ref": "legacy/exact-reference",
                    }
                }
            )
        )
        status, report = _run("--apply", "--mapping", str(mapping))
        assert status == 0
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT credential_ref FROM catalog_database WHERE id = 'vault'"
                    )
                ).scalar_one()
                == "legacy/exact-reference"
            )
    finally:
        engine.dispose()


@pytest.mark.usefixtures("identity_database")
def test_existing_credential_binding_cannot_be_reclassified_as_env(
    tmp_path: Path,
) -> None:
    from sqlalchemy import text

    engine = _legacy_store()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO catalog_database (id, name, credential_ref) VALUES ('vault', 'legacy', 'exact-ref')"
                )
            )
        mapping = tmp_path / "mapping.json"
        mapping.write_text(json.dumps({"vault": {"configuration_origin": "env"}}))
        status, report = _run("--apply", "--mapping", str(mapping))
        assert status == 2
        assert {issue["code"] for issue in report["issues"]} == {"origin_conflict"}
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM catalog_database WHERE display_label IS NOT NULL"
                    )
                ).scalar_one()
                == 0
            )
    finally:
        engine.dispose()


@pytest.mark.usefixtures("identity_database")
def test_shared_sql_does_not_guess_owner_or_rewrite_dependents() -> None:
    from sqlalchemy import text

    engine = _legacy_store()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO catalog_database (id, name, connection) VALUES ('other', 'warehouse', '{}'::jsonb)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO catalog_schema (id, database_id, name) VALUES ('other-schema', 'other', 'public')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO catalog_table (id, schema_id, name) VALUES ('other-table', 'other-schema', 'orders')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO sql_query__table (sql_query_id, table_id) VALUES ('owned-query', 'other-table')"
                )
            )
        status, report = _run("--apply")
        assert status == 0
        assert report["unresolved_sql"]["sql_query"] == ["owned-query", "unknown-query"]
        assert report["unresolved_sql"]["sql_attribute"] == ["attribute"]
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM sql_query WHERE connection_id IS NOT NULL"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM sql_attribute WHERE connection_id IS NOT NULL"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                conn.execute(text("SELECT count(*) FROM sql_query__table")).scalar_one()
                == 2
            )
    finally:
        engine.dispose()
