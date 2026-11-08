# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.connectors.sources import SourceDescriptor


@pytest.fixture
def source() -> SourceDescriptor:
    return SourceDescriptor("a", "Local sales", "sales", "postgres", "managed")


def test_selected_source_is_checked_before_query_setup(
    source: SourceDescriptor, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.retrieval import source_context as context

    resolve = Mock(return_value=source)
    semantic = Mock(return_value=True)
    vectors = Mock()
    monkeypatch.setattr(context, "resolve_source", resolve)
    monkeypatch.setattr(context, "semantic_layer_calculated", semantic)
    monkeypatch.setattr(context, "require_source_vectors_ready", vectors)
    assert context.resolve_query_source(connection_label="Local sales") is source
    resolve.assert_called_once_with(
        connection_id=None, connection_label="Local sales", target_db=None
    )
    semantic.assert_called_once_with(connection_id="a")
    vectors.assert_called_once_with("a")


@pytest.mark.parametrize("failure", ["semantic-empty", "semantic-store", "vectors"])
def test_failed_readiness_is_sanitized_without_selecting_another_source(
    source: SourceDescriptor, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from auto_ontology.retrieval import source_context as context

    resolve = Mock(return_value=source)
    semantic = Mock(return_value=failure != "semantic-empty")
    vectors = Mock()
    if failure == "semantic-store":
        semantic.side_effect = RuntimeError("postgres://private:password@host/store")
    elif failure == "vectors":
        vectors.side_effect = ValueError("secret store error")
    monkeypatch.setattr(context, "resolve_source", resolve)
    monkeypatch.setattr(context, "semantic_layer_calculated", semantic)
    monkeypatch.setattr(context, "require_source_vectors_ready", vectors)
    with pytest.raises(ValueError) as raised:
        context.resolve_query_source(connection_id="a")
    assert raised.value.status_code == 503
    assert raised.value.code == str(raised.value) == "source_not_ready"
    resolve.assert_called_once()
    if failure != "vectors":
        vectors.assert_not_called()


def test_selection_errors_precede_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    from auto_ontology.retrieval import source_context as context
    from auto_ontology.connectors.sources import SourceSelectionError

    error = SourceSelectionError(422, "source_required")
    semantic, vectors = Mock(), Mock()
    monkeypatch.setattr(context, "resolve_source", Mock(side_effect=error))
    monkeypatch.setattr(context, "semantic_layer_calculated", semantic)
    monkeypatch.setattr(context, "require_source_vectors_ready", vectors)
    with pytest.raises(SourceSelectionError) as raised:
        context.resolve_query_source()
    assert raised.value is error
    semantic.assert_not_called()
    vectors.assert_not_called()
