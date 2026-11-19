# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Concept migration preserves every representable legacy edge."""

import psycopg
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config

from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,  # noqa: F401
)
from auto_ontology.infra.postgres import get_postgres_connection_string


@pytest.mark.parametrize(
    "target,cross_source", [("head", False), ("38a47b1ef206", False), ("head", True)]
)
def test_concept_upgrade_preserves_edges_or_rolls_back(
    identity_database,  # noqa: F811
    target,
    cross_source,
):
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "f5b40e9c3271")
    with psycopg.connect(get_postgres_connection_string(), autocommit=True) as db:
        db.execute(
            "INSERT INTO catalog_database (id, name, display_label) VALUES ('a','same','A'), ('b','same','B')"
        )
        db.execute(
            "INSERT INTO catalog_schema (id, database_id, name) VALUES ('sa','a','public'), ('sb','b','public')"
        )
        db.execute(
            "INSERT INTO catalog_table (id, schema_id, name) VALUES ('t','sa','a_table'), ('u','sb','b_table')"
        )
        db.execute(
            "INSERT INTO catalog_column (id, table_id, name, ordinal_position) VALUES ('src','t','ref',1), ('x','t','x',2), ('y','t','y',3), ('z','u','z',1)"
        )
        db.execute("INSERT INTO term (id, name) VALUES ('term','Shared')")
        db.execute(
            "INSERT INTO table__term (table_id, term_id) VALUES ('t','term'), ('u','term')"
        )
        db.execute(
            "INSERT INTO column_attribute (id, name, table_id, source_column, term_name) VALUES ('ax','x','t','x','Shared'), ('ay','y','t','y','Shared'), ('az','z','u','z','Shared')"
        )
        db.execute(
            "INSERT INTO column__has_attribute (column_id, attribute_id) VALUES ('x','ax'), ('y','ay'), ('z','az')"
        )
        db.execute(
            "INSERT INTO column__semantic_fk (column_id, attribute_id, use_in_join_paths) VALUES ('src','ax',true), ('src','ay',false)"
        )
        if cross_source:
            db.execute(
                "INSERT INTO column__semantic_fk (column_id, attribute_id) VALUES ('src','az')"
            )
            with pytest.raises(
                sa.exc.DBAPIError, match=r"src -> az \(cross-source target\)"
            ):
                command.upgrade(config, "head")
            assert db.execute(
                "SELECT attribute_id FROM column__semantic_fk ORDER BY 1"
            ).fetchall() == [("ax",), ("ay",), ("az",)]
            assert db.execute("SELECT version_num FROM alembic_version").fetchone() == (
                "f5b40e9c3271",
            )
        else:
            command.upgrade(config, target)
            if target != "head":
                command.upgrade(config, "head")
            assert db.execute(
                "SELECT kc.column_id, fk.use_in_join_paths FROM column__semantic_fk fk JOIN concept_key__column kc USING (key_id) WHERE fk.column_id='src' ORDER BY 1"
            ).fetchall() == [("x", True), ("y", False)]
            assert db.execute(
                "SELECT database_id FROM term__concept WHERE term_id='term' ORDER BY 1"
            ).fetchall() == [("a",), ("b",)]


@pytest.mark.parametrize("cross_source", [False, True])
def test_upgrade_recognizes_master_concepts(identity_database, cross_source):  # noqa: F811
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, "caee572fa81c")
    with psycopg.connect(get_postgres_connection_string(), autocommit=True) as db:
        # Build the actual master shape using the shared Concept table DDL, then
        # restore the two constraints that distinguish the upstream schema.
        from alembic.migration import MigrationContext
        from alembic.operations import Operations
        from alembic.script import ScriptDirectory

        scripts = ScriptDirectory.from_config(config)
        module = scripts.get_revision("38a47b1ef206").module
        engine = sa.create_engine(
            get_postgres_connection_string().replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        )
        try:
            with (
                engine.begin() as conn,
                Operations.context(MigrationContext.configure(conn)),
            ):
                module.upgrade()
        finally:
            engine.dispose()
        db.execute(
            "ALTER TABLE term__concept DROP COLUMN database_id, ADD CONSTRAINT uq_term__concept_term_id UNIQUE (term_id)"
        )
        db.execute(
            "ALTER TABLE column__semantic_fk DROP CONSTRAINT pk_column__semantic_fk, ADD CONSTRAINT pk_column__semantic_fk PRIMARY KEY (column_id)"
        )
        db.execute("UPDATE alembic_version SET version_num = '0b6f3c2a9d14'")
        db.execute(
            "INSERT INTO catalog_database (id, name) VALUES ('a','shop'), ('b','branch')"
        )
        db.execute(
            "INSERT INTO catalog_schema (id, database_id, name) VALUES ('sa','a','public'), ('sb','b','public')"
        )
        db.execute(
            "INSERT INTO catalog_table (id, schema_id, name) VALUES ('t','sa','items'), ('u','sb','items')"
        )
        db.execute(
            "INSERT INTO catalog_column (id, table_id, name, ordinal_position) VALUES ('src','t','ref',1), ('x','t','x',2)"
        )
        db.execute("INSERT INTO term (id, name) VALUES ('term','Item')")
        db.execute(
            "INSERT INTO table__term (table_id, term_id) VALUES ('t','term'), ('u','term')"
        )
        db.execute(
            "INSERT INTO concept (id, database_id, name, certified) VALUES ('c','a','Curated Item',true)"
        )
        db.execute(
            "INSERT INTO term__concept (term_id, concept_id) VALUES ('term','c')"
        )
        db.execute(
            "INSERT INTO concept_key (id, concept_id, name) VALUES ('key','c','x')"
        )
        db.execute(
            "INSERT INTO concept_key__column (key_id, column_id, table_id) VALUES ('key','x','t')"
        )
        db.execute(
            "INSERT INTO column__semantic_fk (column_id, key_id, use_in_join_paths) VALUES ('src','key',false)"
        )

        command.upgrade(config, "c19a83e47b20")
        assert set(
            db.execute("SELECT version_num FROM alembic_version").fetchall()
        ) == {("0b6f3c2a9d14",), ("c19a83e47b20",)}
        # The operator-owned source backfill is required before final constraints.
        db.execute("UPDATE catalog_database SET display_label = name")
        if cross_source:
            db.execute("UPDATE concept SET database_id = 'b' WHERE id = 'c'")
            with pytest.raises(
                sa.exc.DBAPIError, match="Cannot migrate master Concepts"
            ):
                command.upgrade(config, "head")
            assert set(
                db.execute("SELECT version_num FROM alembic_version").fetchall()
            ) == {("0b6f3c2a9d14",), ("c19a83e47b20",)}
            assert db.execute(
                "SELECT column_id, key_id FROM column__semantic_fk"
            ).fetchall() == [("src", "key")]
            return
        command.upgrade(config, "head")
        assert db.execute("SELECT version_num FROM alembic_version").fetchall() == [
            ("715d9c03a842",)
        ]
        assert db.execute(
            "SELECT column_id, key_id, use_in_join_paths FROM column__semantic_fk"
        ).fetchall() == [("src", "key", False)]
        assert db.execute(
            "SELECT name, certified FROM concept WHERE id = 'c'"
        ).fetchone() == ("Curated Item", True)
        assert db.execute(
            "SELECT database_id FROM term__concept WHERE term_id='term' ORDER BY 1"
        ).fetchall() == [("a",), ("b",)]
        command.check(config)
        with pytest.raises(RuntimeError, match="coordinated backup"):
            command.downgrade(config, "0b6f3c2a9d14")
        with pytest.raises(RuntimeError, match="coordinated backup"):
            command.downgrade(config, "f5b40e9c3271")
        assert db.execute("SELECT version_num FROM alembic_version").fetchall() == [
            ("715d9c03a842",)
        ]
