# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain summary and system prompt loading for seed selection."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field

from auto_ontology.connectors import sources

logger = logging.getLogger(__name__)

_SUMMARY_DIR = Path(".semantic_summaries")
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class DomainSummary(BaseModel):
    """Compact domain context for seed and Term extraction."""

    domains: list[str] = Field(default_factory=list)
    core_entities: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)


def selected_source(connection_id: str) -> sources.SourceDescriptor:
    """Validate runtime identity using metadata only, before any model or probe."""
    for source in sources.list_active_sources():
        if source.connection_id == connection_id:
            return source
    raise ValueError("Semantic compilation requires an active connection_id")


def load_domain_summary(connection_id: str) -> DomainSummary | None:
    path = _SUMMARY_DIR / f"{quote(connection_id, safe='')}.json"
    if not path.exists():
        return None
    try:
        return DomainSummary.model_validate_json(path.read_text())
    except Exception:
        logger.warning("Failed to load domain summary from %s", path)
        return None


def load_system_prompt() -> str:
    path = _PROMPTS_DIR / "system_prompt.txt"
    if path.exists():
        return path.read_text().strip()
    return (
        "You are compiling a business semantic layer over a relational database. "
        "Use clear, human-readable Term names with spaces between words and precise "
        "business language."
    )
