# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source context is checked before SQL dependency extraction."""

from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from auto_ontology.catalog.parsers.sqlglot_extractor import ExtractionResult, TableMatch
from auto_ontology.catalog.sql_parse import parse_query_single


def schema(owner):
    result = MagicMock()
    result.get_db_node.return_value.id = owner
    result.get_db_node.return_value.get_name.return_value = "sales"
    result.get_table_node.return_value.id = f"table-{owner}"
    return result


@patch("auto_ontology.catalog.sql_parse.extract_tables_and_columns")
def test_mixed_source_schema_map_rejected_before_extraction(extract):
    with pytest.raises(ValueError, match="connection_id"):
        parse_query_single(
            "select * from orders",
            {"public": schema("a"), "other": schema("b")},
            ["postgres"],
            connection_id="a",
        )
    extract.assert_not_called()


@pytest.mark.parametrize("database", ["sales", "foreign"])
@patch("auto_ontology.catalog.sql_parse.extract_tables_and_columns")
def test_physical_database_qualifier_must_match_selected_source(extract, database):
    extract.return_value = ExtractionResult(
        tables={"public.orders": TableMatch(schema_name="public")},
        dialect_used="postgres",
    )
    sql = f"select * from {database}.public.orders"
    if database == "foreign":
        with pytest.raises(ValueError, match="database qualifier"):
            parse_query_single(
                sql, {"public": schema("a")}, ["postgres"], connection_id="a"
            )
    else:
        q = parse_query_single(
            sql, {"public": schema("a")}, ["postgres"], connection_id="a"
        )
        assert q.connection_id == "a"
        assert q.sql_node.get_properties()["sql_full_query"] == sql
        assert q.sql_node.get_match_props()["connection_id"] == "a"


@patch("auto_ontology.catalog.write.add_indices")
@patch("auto_ontology.catalog.write.populate_db")
@patch("auto_ontology.catalog.write.populate_queries")
def test_nonempty_history_is_enabled_with_source_context(populate, db, indices):
    from auto_ontology.catalog.write import populate_tabular_data

    history = pd.DataFrame([{"query_text": "select id from orders"}])
    data = {
        "connection_id": "a",
        "database_name": "sales",
        "tables": pd.DataFrame([{"table_name": "orders"}]),
        "columns": pd.DataFrame(),
        "queries": history,
    }
    result = populate_tabular_data(data, 1, "postgres")
    populate.assert_called_once_with(
        db.return_value, history, 1, ["postgres"], connection_id="a"
    )
    assert result is db.return_value
