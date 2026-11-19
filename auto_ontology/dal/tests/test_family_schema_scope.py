# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-local families for shared global Terms, without a Postgres service."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
from unittest.mock import Mock

from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from auto_ontology.dal import schema as s
from auto_ontology.dal import term_families as tf


ROOT = Path(__file__).resolve().parents[3]
REVISION = "f5b40e9c3271"


@pytest.fixture
def database(monkeypatch):
    engine = sa.create_engine("sqlite://")
    with engine.connect() as connection:
        # Only the columns exercised here; production metadata uses Postgres types.
        for ddl in (
            "CREATE TABLE catalog_database (id TEXT PRIMARY KEY, name TEXT)",
            "CREATE TABLE catalog_schema (id TEXT PRIMARY KEY, database_id TEXT)",
            "CREATE TABLE catalog_table (id TEXT PRIMARY KEY, schema_id TEXT)",
            "CREATE TABLE table__term (table_id TEXT, term_id TEXT)",
            "CREATE TABLE column_attribute (id TEXT PRIMARY KEY, table_id TEXT)",
            "CREATE TABLE column_attribute__term (attribute_id TEXT, term_id TEXT)",
            "CREATE TABLE term_family (id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))), schema_id TEXT, prefix TEXT)",
            "CREATE TABLE term__term_family (term_id TEXT, family_id TEXT, base_name TEXT)",
        ):
            connection.exec_driver_sql(ddl)
        for source in ("a", "b"):
            connection.execute(
                s.catalog_database.insert().values(id=source, name="sales")
            )
            connection.execute(
                s.catalog_schema.insert().values(
                    id=f"schema-{source}", database_id=source
                )
            )
            connection.execute(
                s.catalog_table.insert().values(
                    id=f"table-{source}", schema_id=f"schema-{source}"
                )
            )
            connection.execute(
                s.table__term.insert().values(
                    table_id=f"table-{source}", term_id="shared"
                )
            )
            connection.exec_driver_sql(
                "INSERT INTO column_attribute VALUES (?, ?)",
                (f"attribute-{source}", f"table-{source}"),
            )
            connection.execute(
                s.column_attribute__term.insert().values(
                    attribute_id=f"attribute-{source}", term_id="shared"
                )
            )

        def query(statement):
            result = connection.execute(statement)
            return list(result.mappings()) if result.returns_rows else []

        store = Mock(
            query_read=Mock(side_effect=query), query_write=Mock(side_effect=query)
        )
        monkeypatch.setattr(tf, "store", lambda: store)
        monkeypatch.setattr(tf, "write_transaction", nullcontext)
        yield connection, store
    engine.dispose()


def test_metadata_allows_shared_term_in_different_families():
    assert not any(
        isinstance(constraint, sa.UniqueConstraint)
        and list(constraint.columns.keys()) == ["term_id"]
        for constraint in s.term__term_family.constraints
    )


@pytest.mark.parametrize("connection_id", [None, "b"])
def test_prefix_is_from_attributes_owning_schema(database, connection_id):
    connection, _ = database
    connection.exec_driver_sql(
        "INSERT INTO term_family VALUES ('fa', 'schema-a', 'AAA'), ('fb', 'schema-b', 'BBB')"
    )
    connection.exec_driver_sql(
        "INSERT INTO term__term_family VALUES ('shared', 'fa', 'base'), ('shared', 'fb', 'base')"
    )
    statement = sa.select(
        s.column_attribute.c.id,
        tf.attribute_family_prefix(
            s.column_attribute.c.id, connection_id=connection_id
        ).label("prefix"),
    ).order_by(s.column_attribute.c.id)
    assert list(connection.execute(statement)) == [
        ("attribute-a", "AAA" if connection_id is None else None),
        ("attribute-b", "BBB"),
    ]
    compiled = str(statement.compile(dialect=postgresql.dialect()))
    assert "term_family.schema_id = family_owner.schema_id" in compiled
    assert "family_owner.id = family_attribute.table_id" in compiled
    connection.exec_driver_sql("DELETE FROM term__term_family WHERE family_id = 'fb'")
    assert (
        connection.scalar(
            sa.select(tf.attribute_family_prefix(sa.literal("attribute-b")))
        )
        is None
    )


def test_replace_shared_term_keeps_other_sources_family(database):
    connection, _ = database
    for source in ("a", "b", "a"):
        assert (
            tf.replace_term_families(
                "sales",
                [(f"schema-{source}", source, {"shared": "base"})],
                connection_id=source,
            )
            == 1
        )
    assert list(
        connection.execute(
            sa.select(s.term_family.c.schema_id, s.term_family.c.prefix).order_by(
                s.term_family.c.schema_id
            )
        )
    ) == [("schema-a", "a"), ("schema-b", "b")]
    assert tf.fetch_table_families("sales", connection_id="b")[0]["prefix"] == "b"


@pytest.mark.parametrize("connection_id", [None, "a"])
def test_replace_rejects_two_families_for_term_in_same_schema(database, connection_id):
    _, store = database
    with pytest.raises(ValueError, match="one family per schema"):
        tf.replace_term_families(
            "sales",
            [
                ("schema-a", "one", {"shared": "base"}),
                ("schema-a", "two", {"shared": "base"}),
            ],
            connection_id=connection_id,
        )
    store.query_write.assert_not_called()


def test_replace_allows_shared_term_across_schemas(database):
    assert (
        tf.replace_term_families(
            "sales",
            [
                ("schema-a", "a", {"shared": "base"}),
                ("schema-b", "b", {"shared": "base"}),
            ],
        )
        == 2
    )


@pytest.fixture
def migration():
    path = (
        ROOT / "alembic" / "versions" / f"{REVISION}_scope_term_families_to_schemas.py"
    )
    spec = importlib.util.spec_from_file_location("family_scope_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_graph(migration):
    config = Config()
    config.set_main_option("script_location", str(ROOT / "alembic"))
    scripts = ScriptDirectory.from_config(config)
    assert scripts.get_heads() == ["38a47b1ef206"]
    assert scripts.get_revision("38a47b1ef206").down_revision == REVISION
    assert migration.down_revision == "f4a39d8b2160"
    assert scripts.get_revision(REVISION).down_revision == migration.down_revision


def test_migration_upgrade_drops_global_unique(migration, monkeypatch):
    operation = Mock()
    monkeypatch.setattr(migration, "op", operation)
    migration.upgrade()
    operation.drop_constraint.assert_called_once_with(
        "uq_term__term_family_term_id", "term__term_family", type_="unique"
    )


@pytest.mark.parametrize("duplicates", [False, True])
def test_migration_downgrade_guards_existing_memberships(
    migration, monkeypatch, duplicates
):
    operation = Mock()
    operation.get_bind.return_value.execute.return_value.scalar_one.return_value = (
        duplicates
    )
    monkeypatch.setattr(migration, "op", operation)
    if duplicates:
        with pytest.raises(RuntimeError, match="multiple families"):
            migration.downgrade()
        operation.create_unique_constraint.assert_not_called()
    else:
        migration.downgrade()
        operation.create_unique_constraint.assert_called_once_with(
            "uq_term__term_family_term_id", "term__term_family", ["term_id"]
        )
    query = str(operation.get_bind.return_value.execute.call_args.args[0])
    assert "GROUP BY term_id" in query
    assert "HAVING count(*) > 1" in query
    operation.execute.assert_not_called()
    operation.drop_table.assert_not_called()
