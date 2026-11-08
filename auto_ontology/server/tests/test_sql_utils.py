# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation reads selected catalog metadata, never live connectors."""

from unittest.mock import MagicMock, patch

import pytest
import sqlglot

from auto_ontology.server import sql_utils


@pytest.mark.parametrize("dialect", sql_utils._DEFAULT_DIALECTS)
def test_every_default_dialect_is_one_sqlglot_accepts(dialect: str) -> None:
    assert sqlglot.parse_one("SELECT 1", dialect=dialect or None) is not None


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([], sql_utils._DEFAULT_DIALECTS),
        ([{"dialect": None}], sql_utils._DEFAULT_DIALECTS),
        ([{"dialect": "mysql"}], ["mysql"]),
    ],
)
@patch("auto_ontology.server.sql_utils.store")
def test_dialect_comes_from_selected_metadata(
    mock_store: MagicMock, rows, expected
) -> None:
    mock_store.return_value.query_read.return_value = rows
    assert sql_utils.get_dialects("a") == expected
    statement = mock_store.return_value.query_read.call_args.args[0]
    assert statement.compile().params == {"id_1": "a"}


@patch("auto_ontology.server.sql_utils.get_schemas_by_ids")
@patch("auto_ontology.server.sql_utils.fetch_schema_ids_for_connection")
def test_catalog_only_schema_lookup_is_by_id(ids, schemas) -> None:
    ids.return_value = ["s-a"]
    assert sql_utils.get_schemas("a") is schemas.return_value
    ids.assert_called_once_with("a")
    schemas.assert_called_once_with(["s-a"], connection_id="a")


@patch("auto_ontology.server.sql_utils.parse_query_single")
def test_validation_propagates_source_context(parse) -> None:
    sql_utils.validate_sql("select 1", [""], {}, connection_id="a")
    parse.assert_called_once_with(
        sql="select 1", dialects=[""], schemas={}, connection_id="a"
    )
