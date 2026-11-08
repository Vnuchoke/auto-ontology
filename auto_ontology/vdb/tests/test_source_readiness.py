# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from unittest.mock import Mock

import psycopg
from psycopg import sql
import pytest

from auto_ontology.dal.tests.test_connection_identity_schema import (
    identity_database as identity_database,
)
from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb import DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION


@pytest.fixture
def vector_catalog(identity_database: None) -> Iterator[None]:
    with psycopg.connect(get_postgres_connection_string()) as conn:
        conn.execute("CREATE SCHEMA vdb")
        for name in (DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION):
            conn.execute(
                sql.SQL("CREATE TABLE {} (connection_id text)").format(
                    sql.Identifier("vdb", name)
                )
            )
            conn.execute(
                sql.SQL("CREATE INDEX {} ON {} (connection_id)").format(
                    sql.Identifier(f"ix_{name}_connection_id"),
                    sql.Identifier("vdb", name),
                )
            )
            conn.execute(
                sql.SQL("INSERT INTO {} VALUES ('a'), (NULL)").format(
                    sql.Identifier("vdb", name)
                )
            )
    yield


def test_missing_collections_fail_without_creating_schema(
    identity_database: None,
) -> None:
    from auto_ontology.vdb.readiness import require_source_vectors_ready

    with pytest.raises(ValueError, match="source_not_ready"):
        require_source_vectors_ready("a")
    with psycopg.connect(get_postgres_connection_string()) as conn:
        assert conn.execute("SELECT to_regnamespace('vdb')").fetchone() == (None,)


@pytest.mark.usefixtures("vector_catalog")
def test_only_rows_of_selected_source_count() -> None:
    from auto_ontology.vdb.readiness import require_source_vectors_ready

    require_source_vectors_ready("a")
    with pytest.raises(ValueError, match="source_not_ready"):
        require_source_vectors_ready("b")


@pytest.mark.usefixtures("vector_catalog")
@pytest.mark.parametrize("collection", [DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION])
@pytest.mark.parametrize(
    "gap",
    [
        "table",
        "column",
        "index",
        "rows",
        "wrong-index",
        "partial-index",
        "wrong-type",
        "included-column",
    ],
)
def test_either_collection_must_have_indexed_source_rows(
    collection: str, gap: str
) -> None:
    from auto_ontology.vdb.readiness import require_source_vectors_ready

    table = sql.Identifier("vdb", collection)
    with psycopg.connect(get_postgres_connection_string()) as conn:
        if gap == "table":
            conn.execute(sql.SQL("DROP TABLE {}").format(table))
        elif gap == "column":
            conn.execute(
                sql.SQL("ALTER TABLE {} DROP COLUMN connection_id").format(table)
            )
        elif gap == "wrong-type":
            conn.execute(
                sql.SQL(
                    "ALTER TABLE {} ALTER COLUMN connection_id TYPE integer USING NULL"
                ).format(table)
            )
        elif gap in ("index", "wrong-index", "partial-index", "included-column"):
            conn.execute(
                sql.SQL("DROP INDEX {}").format(
                    sql.Identifier("vdb", f"ix_{collection}_connection_id")
                )
            )
            if gap in ("wrong-index", "included-column"):
                conn.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN other text").format(table)
                )
                conn.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} (other)"
                        + (
                            " INCLUDE (connection_id)"
                            if gap == "included-column"
                            else ""
                        )
                    ).format(sql.Identifier(f"ix_{collection}_connection_id"), table)
                )
            elif gap == "partial-index":
                conn.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} (connection_id) WHERE connection_id = 'b'"
                    ).format(sql.Identifier(f"ix_{collection}_connection_id"), table)
                )
        else:
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE connection_id = 'a'").format(table)
            )
    with pytest.raises(ValueError, match="source_not_ready"):
        require_source_vectors_ready("a")


@pytest.mark.parametrize("connection_id", ["", " ", None, 1])
def test_invalid_id_rejected_before_store_access(
    connection_id: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.vdb import readiness

    connect = Mock(side_effect=AssertionError("must validate first"))
    monkeypatch.setattr(readiness.psycopg, "connect", connect)
    with pytest.raises(ValueError, match="source_not_ready"):
        readiness.require_source_vectors_ready(connection_id)
    connect.assert_not_called()


def test_store_failure_never_exposes_connection_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.vdb import readiness

    monkeypatch.setattr(
        readiness, "get_postgres_connection_string", lambda: "synthetic"
    )
    monkeypatch.setattr(
        readiness.psycopg,
        "connect",
        Mock(
            side_effect=psycopg.OperationalError(
                "postgres://private:password@host/store"
            )
        ),
    )
    with pytest.raises(ValueError, match="^source_not_ready$") as raised:
        readiness.require_source_vectors_ready("a")
    assert raised.value.__suppress_context__
