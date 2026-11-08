# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unresolved definitions are inspectable metadata, not LLM context."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from auto_ontology.connectors import sources
from auto_ontology.server.sql_attributes import description_suggester as suggester


@pytest.fixture
def suggestion(monkeypatch):
    row = {
        "id": "attribute",
        "name": "Revenue",
        "term_name": "Shared term",
        "connection_id": "A",
    }
    monkeypatch.setattr(
        suggester, "get_full_sql_attribute_by_id", Mock(return_value=row)
    )
    monkeypatch.setattr(
        sources,
        "list_active_sources",
        Mock(
            return_value=[
                sources.SourceDescriptor("A", "A", "sales", "postgres", "managed")
            ]
        ),
    )
    monkeypatch.setattr(suggester, "get_llm_client", Mock())
    monkeypatch.setattr(
        suggester,
        "invoke_with_structured_output",
        Mock(return_value=SimpleNamespace(description="Total revenue.")),
    )
    monkeypatch.setattr(suggester, "set_sql_attribute_description_suggestion", Mock())
    return row


@pytest.mark.parametrize("connection_id", [None, "inactive"])
def test_unresolved_definition_never_enters_model_prompt(suggestion, connection_id):
    suggestion["connection_id"] = connection_id
    assert suggester.suggest_sql_attribute_description("attribute") is None
    suggester.get_llm_client.assert_not_called()
    suggester.invoke_with_structured_output.assert_not_called()
    suggester.set_sql_attribute_description_suggestion.assert_not_called()


def test_owned_definition_validates_active_metadata_before_model(suggestion):
    def model(**kwargs):
        sources.list_active_sources.assert_called_once_with()
        return "model"

    suggester.get_llm_client.side_effect = model
    assert suggester.suggest_sql_attribute_description("attribute") == "Total revenue."
    suggester.set_sql_attribute_description_suggestion.assert_called_once_with(
        "attribute", "Total revenue."
    )


def test_cached_legacy_suggestion_remains_inspectable_without_model(suggestion):
    suggestion.update(connection_id=None, description_suggestion="Cached description.")
    assert (
        suggester.suggest_sql_attribute_description("attribute")
        == "Cached description."
    )
    suggester.get_llm_client.assert_not_called()
    sources.list_active_sources.assert_not_called()
