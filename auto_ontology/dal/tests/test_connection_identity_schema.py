# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection identity constraints and the operator-controlled migration boundary."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Table, Text, UniqueConstraint

from auto_ontology.dal import schema

ROOT = Path(__file__).resolve().parents[3]
LEGACY = "99d4d343dd0b"
ADDITIVE = "c19a83e47b20"
FINAL = "d27b94f58c31"


def test_root_separates_physical_name_from_required_unique_label() -> None:
    root = schema.catalog_database
    assert not root.c.name.unique
    assert not root.c.display_label.nullable
    assert isinstance(root.c.display_label.type, Text)
    assert root.c.display_label.server_default is None
    assert root.c.configuration_origin.nullable
    assert root.c.credential_ref.nullable
    assert any(
        index.unique
        and str(index.expressions[0]) == "lower(TRIM(BOTH FROM display_label))"
        for index in root.indexes
    )


@pytest.mark.parametrize(
    "table", [schema.sql_query, schema.sql_attribute, schema.custom_analysis]
)
def test_executable_metadata_has_nullable_explicit_source_ownership(
    table: Table,
) -> None:
    owner = table.c.connection_id
    assert owner.nullable and owner.server_default is None and owner.default is None
    assert next(iter(owner.foreign_keys)).column is schema.catalog_database.c.id
    assert not any(
        isinstance(constraint, UniqueConstraint)
        and [column.name for column in constraint.columns] == ["name"]
        for constraint in table.constraints
    )
    if table is schema.sql_query:
        assert any(
            index.unique
            and [str(expression) for expression in index.expressions]
            == ["sql_query.connection_id", "md5(sql_full_query)"]
            for index in table.indexes
        )
    else:
        assert any(
            isinstance(constraint, UniqueConstraint)
            and [column.name for column in constraint.columns]
            == ["connection_id", "name"]
            for constraint in table.constraints
        )


def test_prisma_preserves_frontend_ownership_and_unbound_legacy_conversations() -> None:
    prisma = (ROOT / "frontend/prisma/schema.prisma").read_text()
    conversation = prisma.split("model Conversation {", 1)[1].split("\n}", 1)[0]
    field = next(
        line.split() for line in conversation.splitlines() if "connection_id" in line
    )
    assert field == ["connection_id", "String?"]
    assert 'schemas  = ["frontend"]' in prisma
    assert '"public"' not in prisma


@pytest.fixture
def identity_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if os.environ.get("CONNECTION_IDENTITY_MIGRATION_TEST") != "1":
        pytest.skip("requires an explicitly authorized disposable migration database")
    assert os.environ.get("PYTHON_DOTENV_DISABLED") == "1"
    assert os.environ.get("POSTGRES_HOST") in {"127.0.0.1", "localhost"}

    import psycopg
    from psycopg import sql

    from auto_ontology.infra.postgres import get_postgres_connection_string

    # Alembic fileConfig disables existing loggers; do not leak that into other tests.
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            monkeypatch.setattr(logger, "disabled", logger.disabled)

    database = f"identity_schema_{uuid4().hex}"
    with psycopg.connect(get_postgres_connection_string(), autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        monkeypatch.setenv("POSTGRES_DATABASE", database)
        try:
            yield
        finally:
            admin.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )


def test_staged_upgrade_preserves_legacy_and_enforces_source_constraints(
    identity_database: None,
) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError

    from auto_ontology.dal.session import sqlalchemy_url

    config = Config(str(ROOT / "alembic.ini"))
    engine = create_engine(sqlalchemy_url())
    command.upgrade(config, LEGACY)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO catalog_database (id, name, description) VALUES ('legacy-root', 'sales', 'curated root')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO catalog_schema (id, database_id, name, description) VALUES ('legacy-schema', 'legacy-root', 'public', 'curated schema')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO catalog_table (id, schema_id, name, description) VALUES ('legacy-table', 'legacy-schema', 'orders', 'curated table')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO catalog_column (id, table_id, name, description) VALUES ('legacy-column', 'legacy-table', 'id', 'curated column')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sql_query (id, sql_full_query) VALUES ('legacy-query', 'SELECT 1')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sql_attribute (id, name) VALUES ('legacy-attribute', 'revenue')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO custom_analysis (id, name) VALUES ('legacy-analysis', 'revenue')"
            )
        )
    command.upgrade(config, ADDITIVE)
    with engine.connect() as conn:
        assert conn.execute(
            text(
                "SELECT display_label, configuration_origin, credential_ref FROM catalog_database WHERE id = 'legacy-root'"
            )
        ).one() == (None, None, None)
        for table in ("sql_query", "sql_attribute", "custom_analysis"):
            assert (
                conn.execute(text(f"SELECT connection_id FROM {table}")).scalar_one()
                is None
            )
    # Finalization must stop rather than invent a label or an owner.
    with pytest.raises(Exception, match="(?i)(backfill|display_label)"):
        command.upgrade(config, FINAL)
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE catalog_database SET display_label = 'Sales local', configuration_origin = 'managed', credential_ref = 'legacy/exact/sales' WHERE id = 'legacy-root'"
            )
        )
        conn.execute(
            text(
                "UPDATE sql_query SET connection_id = 'legacy-root' WHERE id = 'legacy-query'"
            )
        )
    command.upgrade(config, FINAL)
    with engine.connect() as conn:
        for table, row_id, description in (
            ("catalog_database", "legacy-root", "curated root"),
            ("catalog_schema", "legacy-schema", "curated schema"),
            ("catalog_table", "legacy-table", "curated table"),
            ("catalog_column", "legacy-column", "curated column"),
        ):
            assert (
                conn.execute(
                    text(f"SELECT description FROM {table} WHERE id = :id"),
                    {"id": row_id},
                ).scalar_one()
                == description
            )
        assert (
            conn.execute(
                text(
                    "SELECT credential_ref FROM catalog_database WHERE id = 'legacy-root'"
                )
            ).scalar_one()
            == "legacy/exact/sales"
        )
        assert (
            conn.execute(
                text(
                    "SELECT connection_id FROM sql_attribute WHERE id = 'legacy-attribute'"
                )
            ).scalar_one()
            is None
        )
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO catalog_database (id, name, display_label) VALUES ('second-root', 'sales', 'Sales prod')"
            )
        )
        for table, columns, values in (
            ("sql_query", "sql_full_query", "'SELECT 1'"),
            ("sql_attribute", "name", "'revenue'"),
            ("custom_analysis", "name", "'revenue'"),
        ):
            conn.execute(
                text(
                    f"INSERT INTO {table} (connection_id, {columns}) VALUES ('second-root', {values})"
                )
            )
            conn.execute(
                text(
                    f"INSERT INTO {table} (connection_id, {columns}) VALUES ('legacy-root', {values}) ON CONFLICT DO NOTHING"
                )
            )
    for statement in (
        "INSERT INTO catalog_database (name, display_label) VALUES ('other', ' sales LOCAL ')",
        "INSERT INTO catalog_database (name, display_label) VALUES ('other', '   ')",
        "INSERT INTO catalog_database (name, display_label, configuration_origin) VALUES ('other', 'Other', 'unknown')",
        "INSERT INTO sql_query (connection_id, sql_full_query) VALUES ('second-root', 'SELECT 1')",
        "INSERT INTO sql_attribute (connection_id, name) VALUES ('second-root', 'revenue')",
        "INSERT INTO custom_analysis (connection_id, name) VALUES ('second-root', 'revenue')",
        "INSERT INTO sql_query (connection_id, sql_full_query) VALUES ('missing-root', 'SELECT 2')",
    ):
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(text(statement))
    # Downgrade cannot merge or delete the newly distinct physical sources.
    with pytest.raises(Exception, match="(?i)(duplicate|unique|rollback|backup)"):
        command.downgrade(config, ADDITIVE)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM catalog_database WHERE name = 'sales'")
            ).scalar_one()
            == 2
        )
        assert (
            conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            == FINAL
        )
    engine.dispose()
