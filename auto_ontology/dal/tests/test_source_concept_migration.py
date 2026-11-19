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
    identity_database,  # noqa: F401
)
from auto_ontology.infra.postgres import get_postgres_connection_string


@pytest.mark.parametrize("cross_source", [False, True])
def test_concept_upgrade_preserves_edges_or_rolls_back(identity_database, cross_source):  # noqa: F811
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
            command.upgrade(config, "head")
            assert db.execute(
                "SELECT kc.column_id, fk.use_in_join_paths FROM column__semantic_fk fk JOIN concept_key__column kc USING (key_id) WHERE fk.column_id='src' ORDER BY 1"
            ).fetchall() == [("x", True), ("y", False)]
            assert db.execute(
                "SELECT database_id FROM term__concept WHERE term_id='term' ORDER BY 1"
            ).fetchall() == [("a",), ("b",)]
