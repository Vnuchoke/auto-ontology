# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the entity-coverage LangGraph pipeline."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from auto_ontology.retrieval.data_access.custom_analyses import fetch_custom_analyses
from auto_ontology.retrieval.entity_coverage.graph import create_graph
from auto_ontology.retrieval.entity_coverage.state import (
    DEFAULT_MAX_DISTANCE,
    EntityCoveragePayload,
)
from auto_ontology.retrieval.text_to_sql.state import AgentState
from auto_ontology.utils.llm_invoke import get_llm_client
from auto_ontology.connectors.sources import SourceDescriptor
from auto_ontology.retrieval.source_context import resolve_query_source

logger = logging.getLogger(__name__)

try:
    llm_client = get_llm_client()
except (ValueError, EnvironmentError, OSError) as exc:
    logger.error("Failed to initialize LLM client: %s", exc)
    llm_client = None

graph = create_graph()
app = graph.compile()


def _build_state(
    payload: EntityCoveragePayload, source: SourceDescriptor
) -> AgentState:
    connectors = [
        connector
        for connector in payload.get("connectors", [])
        if getattr(connector, "connection_id", None) == source.connection_id
    ]
    if not connectors:
        raise ValueError(
            "EntityCoveragePayload is missing required 'connectors'. "
            "Provide a non-empty list of database connectors."
        )

    data_retriever = payload.get("data_retriever")
    if data_retriever is None:
        raise ValueError("EntityCoveragePayload is missing required 'data_retriever'.")

    semantic_retriever = payload.get("semantic_retriever")
    if semantic_retriever is None:
        logger.warning(
            "No 'semantic_retriever' in payload — candidate searches will be skipped."
        )

    acronyms = payload.get("acronyms", [])
    domain_rules = fetch_custom_analyses(connection_id=source.connection_id) + list(
        acronyms or []
    )

    initial_path_state = dict(payload.get("path_state") or {})
    max_distance = payload.get("max_distance", DEFAULT_MAX_DISTANCE)
    initial_path_state["max_distance"] = float(max_distance)
    initial_path_state["return_uncovered_entities"] = bool(
        payload.get("return_uncovered_entities", False)
    )

    initial_path_state["connection_id"] = source.connection_id
    initial_path_state["target_db"] = source.database_name
    provenance = {
        "connection_id": source.connection_id,
        "display_label": source.display_label,
        "database_name": source.database_name,
    }

    state: dict[str, Any] = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "connectors": connectors,
        "messages": [HumanMessage(content=payload["question"])],
        "path_state": initial_path_state,
        "data_retriever": data_retriever,
        "semantic_retriever": semantic_retriever,
        "decision": "",
        "domain_rules": domain_rules,
        "glossary": list(acronyms or []),
        "connection_id": source.connection_id,
        "source": provenance,
    }
    return state  # type: ignore[return-value]


def get_coverage_response(payload: EntityCoveragePayload) -> dict:
    """Run the entity-coverage graph and return coverage and ranked candidates."""
    source = resolve_query_source(
        connection_id=payload.get("connection_id"),
        connection_label=payload.get("connection_label"),
        target_db=payload.get("target_db"),
    )
    if llm_client is None:
        raise RuntimeError("LLM client is not configured.")

    state = _build_state(payload, source)
    final_state = app.invoke(state, config={"recursion_limit": 10})

    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")
    if isinstance(final_response, dict):
        return {**final_response, "source": state["source"]}
    return {
        "coverage": 0.0,
        "candidates": [],
        "source": state["source"],
    }


__all__ = [
    "get_coverage_response",
    "app",
    "graph",
    "llm_client",
]
