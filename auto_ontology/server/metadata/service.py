# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration for ``question-entity-coverage``.

``entity_coverage`` runs a fast LangGraph that extracts entities, retrieves
semantic candidates, and returns a deterministic coverage grade.

The pipeline shares retriever/connector state and is not safe to run in parallel
(the same constraint that makes chat single-slot), so runs are serialized here.
"""

from __future__ import annotations

import logging
import threading

from auto_ontology.retrieval.entity_coverage.main import get_coverage_response
from auto_ontology.retrieval.entity_coverage.main import (
    llm_client as coverage_llm_client,
)
from auto_ontology.retrieval.entity_coverage.state import (
    DEFAULT_MAX_DISTANCE,
    EntityCoveragePayload,
)
from auto_ontology.server.chat.settings_dal import fetch_acronyms, fetch_custom_prompts
from auto_ontology.connectors.registry import get_connectors_for_subject_token
from auto_ontology.connectors.sources import SourceDescriptor, SourceSelectionError
from auto_ontology.connectors.databricks_oauth import DatabricksOAuthError
from auto_ontology.retrieval.source_context import resolve_query_source
from auto_ontology.utils.retriever import (
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

logger = logging.getLogger(__name__)

# Retrieval runs share retriever/connector state and a single LLM budget and are
# not safe to run in parallel; serialize like the chat pool's single slot.
_run_lock = threading.Lock()


class PredictionFlowError(RuntimeError):
    """The prediction flow could not produce a result for the question."""


def _build_coverage_payload(
    question: str,
    source: SourceDescriptor,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    *,
    subject_token: str | None = None,
) -> EntityCoveragePayload:
    """Assemble the payload for the entity-coverage pipeline."""
    try:
        connectors = get_connectors_for_subject_token(
            subject_token, connection_id=source.connection_id
        )
    except (SourceSelectionError, DatabricksOAuthError):
        raise
    except Exception:
        raise SourceSelectionError(503, "source_unavailable") from None
    if not connectors:
        raise SourceSelectionError(503, "source_unavailable")
    custom_prompts = fetch_custom_prompts()
    return {
        "question": question,
        "data_retriever": get_data_objects_retriever(),
        "semantic_retriever": get_semantic_objects_retriever(),
        "connectors": connectors,
        "acronyms": fetch_acronyms(),
        "custom_prompts": custom_prompts,
        "max_distance": max_distance,
        "connection_id": source.connection_id,
    }


def entity_coverage(
    question: str,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    *,
    connection_id: str | None = None,
    connection_label: str | None = None,
    target_db: str | None = None,
    subject_token: str | None = None,
) -> dict:
    """Return ranked semantic candidates and a 0–1 entity coverage grade.

    Raises :class:`PredictionFlowError` when the flow cannot produce a result.
    """
    source = resolve_query_source(
        connection_id=connection_id,
        connection_label=connection_label,
        target_db=target_db,
    )
    if coverage_llm_client is None:
        raise PredictionFlowError("LLM client is not configured.")
    with _run_lock:
        try:
            return get_coverage_response(
                _build_coverage_payload(
                    question,
                    source,
                    max_distance=max_distance,
                    subject_token=subject_token,
                )
            )
        except (SourceSelectionError, DatabricksOAuthError):
            raise
        except (ValueError, RuntimeError) as exc:
            raise PredictionFlowError(str(exc)) from exc
