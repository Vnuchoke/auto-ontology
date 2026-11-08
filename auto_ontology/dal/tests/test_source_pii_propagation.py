# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

from auto_ontology.dal import pii, schema as s
from auto_ontology.dal.session import dispose_engine, store
from auto_ontology.dal.tests.test_connection_identity_schema import (
    ROOT,
    identity_database as identity_database,
)


@pytest.fixture
def pii_sources(identity_database: None) -> Iterator[None]:
    dispose_engine()
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        db = store()
        db.query_write(s.tag.insert().values(id="pii", name="PII"))
        db.query_write(s.term.insert().values(id="shared", name="Shared"))
        for cid in ("a", "b"):
            db.query_write(
                s.catalog_database.insert().values(
                    id=cid, name="sales", display_label=cid
                )
            )
            db.query_write(
                s.catalog_schema.insert().values(
                    id=f"s-{cid}", name="public", database_id=cid
                )
            )
            db.query_write(
                s.catalog_table.insert().values(
                    id=f"t-{cid}", name="orders", schema_id=f"s-{cid}"
                )
            )
            db.query_write(
                s.catalog_column.insert().values(
                    id=f"c-{cid}", name="email", table_id=f"t-{cid}"
                )
            )
            db.query_write(
                s.column_attribute.insert().values(
                    id=f"attr-{cid}",
                    name=f"Email {cid}",
                    term_name="Shared",
                    source_column="email",
                    table_id=f"t-{cid}",
                )
            )
            db.query_write(
                s.column_attribute__term.insert().values(
                    attribute_id=f"attr-{cid}", term_id="shared"
                )
            )
            db.query_write(
                s.column__has_attribute.insert().values(
                    column_id=f"c-{cid}", attribute_id=f"attr-{cid}"
                )
            )
            db.query_write(
                s.sql_attribute.insert().values(
                    id=f"sql-{cid}", name="Email", connection_id=cid
                )
            )
            db.query_write(
                s.sql_attribute__term.insert().values(
                    attribute_id=f"sql-{cid}", term_id="shared"
                )
            )
            db.query_write(
                s.sql_query.insert().values(
                    id=f"q-{cid}",
                    connection_id=cid,
                    sql_full_query="SELECT email FROM orders",
                )
            )
            db.query_write(
                s.sql_attribute__sql.insert().values(
                    attribute_id=f"sql-{cid}", sql_query_id=f"q-{cid}"
                )
            )
            db.query_write(
                s.sql_query__column.insert().values(
                    sql_query_id=f"q-{cid}", column_id=f"c-{cid}"
                )
            )
            db.query_write(
                s.sql_query__table.insert().values(
                    sql_query_id=f"q-{cid}", table_id=f"t-{cid}"
                )
            )
            db.query_write(
                s.tag_target.insert().values(tag_id="pii", column_id=f"c-{cid}")
            )
        yield
    finally:
        dispose_engine()


def _snapshot_b() -> list[dict]:
    return store().query_read(
        select(s.column_attribute).where(s.column_attribute.c.id == "attr-b")
    ) + store().query_read(
        select(s.sql_attribute).where(s.sql_attribute.c.id == "sql-b")
    )


@pytest.mark.usefixtures("pii_sources")
def test_source_propagation_does_not_change_same_named_neighbor() -> None:
    from auto_ontology.pii_detection.service import propagate_pii_to_attributes

    before = _snapshot_b()
    result = propagate_pii_to_attributes(connection_id="a")
    assert _snapshot_b() == before
    assert (result.column_attributes, result.sql_attributes) == (1, 1)
    targets = store().query_read(
        select(
            s.tag_target.c.column_attribute_id, s.tag_target.c.sql_attribute_id
        ).where(s.tag_target.c.column_id.is_(None))
    )
    assert {r["column_attribute_id"] or r["sql_attribute_id"] for r in targets} == {
        "attr-a",
        "sql-a",
    }
    assert pii.tag_attributes_of_tagged_columns("pii", connection_id="a") == (0, 0)
    assert pii.tag_attributes_of_tagged_columns("pii", connection_id="b") == (1, 1)


@pytest.mark.usefixtures("pii_sources")
def test_foreign_sql_dependencies_are_not_scoped_pii_candidates() -> None:
    store().query_write(
        s.sql_query__column.insert().values(sql_query_id="q-a", column_id="c-b")
    )
    assert pii.tag_attributes_of_tagged_columns("pii", connection_id="a") == (1, 0)
    assert store().query_read(
        select(s.sql_attribute.c.pii_processed).where(s.sql_attribute.c.id == "sql-a")
    ) == [{"pii_processed": False}]
