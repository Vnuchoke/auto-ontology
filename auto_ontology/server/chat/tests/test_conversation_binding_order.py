# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest

from auto_ontology.connectors.sources import SourceSelectionError
from auto_ontology.retrieval import source_context
from auto_ontology.server.chat import conversation_dal as dal
from auto_ontology.server.chat.tests.test_conversation_source_binding import (
    A,
    B,
    _seed,
    _snapshot,
    identity_database as identity_database,
)


def test_bound_conflict_precedes_other_sources_readiness(
    identity_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = uuid4()
    _seed(conversation, source=A)
    before = _snapshot()
    readiness = Mock(return_value=False)
    monkeypatch.setattr(source_context, "semantic_layer_calculated", readiness)
    with pytest.raises(SourceSelectionError) as error:
        dal.prepare_conversation(
            conversation_id=conversation,
            user_id="owner",
            question="wrong source",
            source="app",
            connection_id=B,
        )
    assert (error.value.status_code, error.value.code) == (409, "source_conflict")
    readiness.assert_not_called()
    assert _snapshot() == before
