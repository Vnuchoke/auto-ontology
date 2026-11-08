# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source selection must precede SQL/model work and never guess by name."""

from unittest.mock import Mock
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from auto_ontology.connectors import sources
from auto_ontology.server.sql_attributes import service as attrs
from auto_ontology.server.custom_analyses import service as analyses
from auto_ontology.catalog.model import CatalogNode


@pytest.fixture
def active_sources(monkeypatch):
    descriptors = [
        sources.SourceDescriptor(cid, cid, "sales", "postgres", "managed")
        for cid in ("A", "B")
    ]
    monkeypatch.setattr(sources, "list_active_sources", lambda: descriptors)
    return descriptors


@pytest.mark.parametrize("service", [attrs, analyses])
@pytest.mark.parametrize(
    "selector, error", [(None, "source_required"), ("C", "invalid_source_selector")]
)
def test_selection_fails_before_sql_or_conflict_checks(
    monkeypatch, active_sources, service, selector, error
):
    validate = Mock(side_effect=AssertionError("SQL must not run"))
    monkeypatch.setattr(service, "validate_sql", validate)
    with pytest.raises(ValueError, match=error):
        if service is attrs:
            attrs.validate_sql_attribute(expression="SELECT 1", connection_id=selector)
        else:
            analyses.validate_custom_analysis_sql("SELECT 1", connection_id=selector)
    validate.assert_not_called()


@pytest.mark.parametrize("service", [attrs, analyses])
def test_selected_id_not_physical_name_reaches_sql(
    monkeypatch, active_sources, service
):
    dialects, schemas, validate = (
        Mock(return_value=["postgres"]),
        Mock(return_value={}),
        Mock(),
    )
    monkeypatch.setattr(service, "get_dialects", dialects)
    monkeypatch.setattr(service, "get_schemas", schemas)
    monkeypatch.setattr(service, "validate_sql", validate)
    if service is attrs:
        attrs.validate_sql_attribute(expression="SELECT 1", connection_id="B")
    else:
        analyses.validate_custom_analysis_sql("SELECT 1", connection_id="B")
    dialects.assert_called_once_with("B")
    schemas.assert_called_once_with("B")
    assert validate.call_args.kwargs == {"connection_id": "B"}


@pytest.fixture
def writes(monkeypatch, active_sources):
    """Small store spy: a failed relink must restore detached metadata."""
    state = {"linked": True, "transactions": 0, "rolled_back": False}

    @contextmanager
    def transaction():
        before = state["linked"]
        state["transactions"] += 1
        try:
            yield
        except Exception:
            state["linked"] = before
            state["rolled_back"] = True
            raise

    state["add"] = Mock()
    for service in (attrs, analyses):
        monkeypatch.setattr(service, "write_transaction", transaction)
        monkeypatch.setattr(service, "add_query", state["add"])
        monkeypatch.setattr(
            service, "detach_existing_sql_edges", lambda _: state.update(linked=False)
        )
        monkeypatch.setattr(service, "get_dialects", Mock(return_value=["postgres"]))
        monkeypatch.setattr(service, "get_schemas", Mock(return_value={}))

    def query(sql, *_, connection_id):
        result = SimpleNamespace(
            connection_id=connection_id,
            edges=[],
            sql_node=CatalogNode(
                "sql", "Sql", {"sql_full_query": sql, "connection_id": connection_id}
            ),
        )
        result.get_edges = lambda: result.edges
        return result

    for service in (attrs, analyses):
        monkeypatch.setattr(service, "validate_sql", Mock(side_effect=query))
    monkeypatch.setattr(attrs, "find_attr_by_name", Mock(return_value=None))
    monkeypatch.setattr(attrs, "find_attr_by_expression", Mock(return_value=None))
    monkeypatch.setattr(
        attrs, "get_slim_term_by_id", Mock(return_value={"name": "Shared term"})
    )
    monkeypatch.setattr(
        attrs,
        "get_full_sql_attribute_by_id",
        Mock(
            return_value={
                "id": "definition",
                "name": "Orders",
                "connection_id": "B",
                "expression": "SELECT old",
            }
        ),
    )
    monkeypatch.setattr(attrs, "link_to_term", Mock())
    monkeypatch.setattr(attrs, "clear_sql_attribute_description_suggestion", Mock())
    monkeypatch.setattr(attrs.sql_attr_dal, "update_sql_attribute", Mock())
    monkeypatch.setattr(attrs, "get_tag_by_name", Mock(return_value=None))
    monkeypatch.setattr(attrs, "realign_sql_attribute_pii", Mock())
    monkeypatch.setattr(attrs, "_reembed_sql_attribute", Mock())
    monkeypatch.setattr(attrs, "_embed_sql_attribute", Mock())
    monkeypatch.setattr(attrs, "get_semantic_vdb", Mock())
    monkeypatch.setattr(attrs, "get_embed_params", Mock())
    monkeypatch.setattr(analyses, "find_analysis_by_name", Mock(return_value=None))
    monkeypatch.setattr(analyses, "find_analysis_by_sql", Mock(return_value=None))
    monkeypatch.setattr(
        analyses, "get_custom_analysis_by_id", Mock(return_value="definition")
    )
    monkeypatch.setattr(
        analyses.analysis_dal,
        "get_custom_analysis_connection_id",
        Mock(return_value="B"),
    )
    monkeypatch.setattr(analyses, "_embed_analysis", Mock())
    monkeypatch.setattr("auto_ontology.vdb.get_semantic_vdb", Mock())
    return state


def update(service, connection_id=None):
    if service is attrs:
        return attrs.update_sql_attribute(
            attr_id="definition",
            name="Orders",
            description="Count",
            expression="SELECT new",
            term_id="term",
            connection_id=connection_id,
        )
    return analyses.update_custom_analysis(
        "definition", "Orders", "Count", "SELECT new", connection_id=connection_id
    )


@pytest.mark.parametrize("service", [attrs, analyses])
def test_update_inherits_source_and_owns_nodes(writes, service):
    row = update(service)
    assert row["connection_id"] == "B"
    assert row["database_name"] == "sales"
    assert service.validate_sql.call_args.kwargs["connection_id"] == "B"
    edges = writes["add"].call_args.args[0]
    definition, sql_node, _ = edges[-1]
    assert definition.props["connection_id"] == "B"
    assert definition.match_props["connection_id"] == "B"
    assert sql_node.match_props == {
        "sql_full_query": "SELECT new",
        "connection_id": "B",
    }
    assert writes["add"].call_args.kwargs == {"connection_id": "B"}


@pytest.mark.parametrize("service", [attrs, analyses])
def test_update_rejects_switch_before_validation_or_detach(writes, service):
    with pytest.raises(ValueError, match="source_conflict"):
        update(service, "A")
    service.validate_sql.assert_not_called()
    assert writes["linked"] is True
    assert writes["transactions"] == 0


@pytest.mark.parametrize("service", [attrs, analyses])
def test_failed_add_query_rolls_back_detach(writes, service):
    writes["add"].side_effect = ValueError("foreign-source dependency")
    with pytest.raises(ValueError, match="foreign-source"):
        update(service)
    assert writes["transactions"] == 1
    assert writes["rolled_back"] is True
    assert writes["linked"] is True


def test_attribute_update_expression_conflict_is_source_scoped(writes):
    attrs.find_attr_by_expression.return_value = {"id": "other"}
    with pytest.raises(attrs.SqlAttributeExpressionConflict):
        update(attrs)
    assert attrs.find_attr_by_expression.call_args.kwargs == {
        "term_id": "term",
        "expression": "SELECT new",
        "exclude_id": "definition",
        "connection_id": "B",
    }
    assert writes["linked"] is True


def create(service, connection_id=None):
    if service is attrs:
        return attrs.create_sql_attribute(
            name="Orders",
            description="Count",
            expression="SELECT new",
            term_id="term",
            connection_id=connection_id,
        )
    return analyses.create_custom_analysis(
        "Orders", "Count", "SELECT new", connection_id=connection_id, embed=False
    )


@pytest.mark.parametrize("service", [attrs, analyses])
def test_create_conflicts_and_natural_keys_use_selected_id(writes, service):
    rows = [create(service, cid) for cid in ("A", "B")]
    assert [row["connection_id"] for row in rows] == ["A", "B"]
    assert [row["database_name"] for row in rows] == ["sales", "sales"]
    assert rows[0]["id"] != rows[1]["id"]
    name_check = (
        attrs.find_attr_by_name if service is attrs else analyses.find_analysis_by_name
    )
    expression_check = (
        attrs.find_attr_by_expression
        if service is attrs
        else analyses.find_analysis_by_sql
    )
    assert [call.kwargs["connection_id"] for call in name_check.call_args_list] == [
        "A",
        "B",
    ]
    assert [
        call.kwargs["connection_id"] for call in expression_check.call_args_list
    ] == ["A", "B"]
    for call, cid in zip(writes["add"].call_args_list, ("A", "B"), strict=True):
        definition, sql_node, _ = call.args[0][-1]
        assert definition.props["connection_id"] == cid
        assert definition.match_props == {"name": "Orders", "connection_id": cid}
        assert sql_node.match_props["connection_id"] == cid


@pytest.mark.parametrize("service", [attrs, analyses])
@pytest.mark.parametrize(
    "selector, error",
    [
        (None, "source_required"),
        ("C", "invalid_source_selector"),
        ("", "invalid_source_selector"),
    ],
)
def test_invalid_create_never_checks_conflicts_or_mutates(
    writes, service, selector, error
):
    with pytest.raises(ValueError, match=error):
        create(service, selector)
    service.validate_sql.assert_not_called()
    service.get_dialects.assert_not_called()
    service.get_schemas.assert_not_called()
    name_check = (
        attrs.find_attr_by_name if service is attrs else analyses.find_analysis_by_name
    )
    name_check.assert_not_called()
    writes["add"].assert_not_called()
    assert writes["transactions"] == 0


@pytest.mark.parametrize("service", [attrs, analyses])
def test_single_source_can_be_chosen_automatically(writes, active_sources, service):
    active_sources.pop()
    assert create(service)["connection_id"] == "A"


@pytest.mark.parametrize("service", [attrs, analyses])
@pytest.mark.parametrize("stored_id", [None, "deleted-source"])
def test_unowned_or_inactive_definition_cannot_execute(writes, service, stored_id):
    if service is attrs:
        attrs.get_full_sql_attribute_by_id.return_value["connection_id"] = stored_id
    else:
        analyses.analysis_dal.get_custom_analysis_connection_id.return_value = stored_id
    with pytest.raises(ValueError, match="source_required|invalid_source_selector"):
        update(service)
    service.validate_sql.assert_not_called()
    assert writes["linked"] is True
    assert writes["transactions"] == 0


@pytest.mark.parametrize("service", [attrs, analyses])
def test_missing_definition_keeps_none_semantics(writes, service):
    if service is attrs:
        attrs.get_full_sql_attribute_by_id.return_value = None
    else:
        analyses.get_custom_analysis_by_id.return_value = None
    assert update(service, "A") is None
    service.validate_sql.assert_not_called()


def test_internal_connector_adapter_rejects_same_name_ambiguity(writes, active_sources):
    with pytest.raises(ValueError, match="invalid_source_selector"):
        attrs.validate_sql_attribute(expression="SELECT 1", connector="sales")
    attrs.validate_sql.assert_not_called()
    active_sources.pop()
    attrs.validate_sql_attribute(expression="SELECT 1", connector="sales")
    assert attrs.validate_sql.call_args.kwargs["connection_id"] == "A"


@pytest.mark.parametrize("service", [attrs, analyses])
def test_validate_existing_inherits_source(writes, service):
    if service is attrs:
        attrs.validate_sql_attribute(expression="SELECT 1", attribute_id="definition")
    else:
        analyses.validate_custom_analysis_sql("SELECT 1", analysis_id="definition")
    assert service.validate_sql.call_args.kwargs["connection_id"] == "B"
