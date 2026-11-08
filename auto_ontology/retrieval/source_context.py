# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an active source and prove minimum readiness before query work."""

from auto_ontology.connectors.sources import (
    SourceDescriptor,
    SourceSelectionError,
    resolve_source,
)
from auto_ontology.dal.terms import semantic_layer_calculated
from auto_ontology.vdb.readiness import require_source_vectors_ready


def resolve_query_source(
    *,
    connection_id: str | None = None,
    connection_label: str | None = None,
    target_db: str | None = None,
) -> SourceDescriptor:
    """Preserve selector errors; report unavailable metadata/indexes without secrets."""
    source = resolve_source(
        connection_id=connection_id,
        connection_label=connection_label,
        target_db=target_db,
    )
    try:
        if not semantic_layer_calculated(connection_id=source.connection_id):
            raise ValueError("source_not_ready")
        require_source_vectors_ready(source.connection_id)
    except Exception:
        raise SourceSelectionError(503, "source_not_ready") from None
    return source
