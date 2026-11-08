# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Access Layer — CustomAnalysis write orchestration.

All direct store access lives in auto_ontology/dal/custom_analyses.py.
This module only keeps orchestration: SQL validation, node
persistence, and VDB embedding — the three concerns that can't be
cleanly separated into a pure-graph layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from auto_ontology.catalog.store.queries import add_query
from auto_ontology.catalog.model import CatalogNode
from auto_ontology.catalog.constants import (
    Labels,
    Props,
)
from auto_ontology.connectors import sources
from auto_ontology.dal import custom_analyses as analysis_dal
from auto_ontology.dal.session import write_transaction
from auto_ontology.dal.custom_analyses import (
    CustomAnalysisNameConflict,
    CustomAnalysisSqlConflict,
    CustomAnalysisSqlError,
    count_custom_analyses,
    delete_custom_analysis_node,
    detach_existing_sql_edges,
    embed_custom_analyses,
    fetch_database_name_for_analysis,
    find_analysis_by_name,
    find_analysis_by_sql,
    get_custom_analysis_by_id,
    list_custom_analyses,
)
from auto_ontology.server.sql_utils import get_dialects, get_schemas, validate_sql

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "CustomAnalysisNameConflict",
    "CustomAnalysisSqlConflict",
    "CustomAnalysisSqlError",
    "count_custom_analyses",
    "list_custom_analyses",
    "validate_custom_analysis_sql",
    "create_custom_analysis",
    "update_custom_analysis",
    "delete_custom_analysis",
]


# ---------------------------------------------------------------------------
# Validation API
# ---------------------------------------------------------------------------


def _resolve_source(connection_id: str | None) -> sources.SourceDescriptor:
    """Choose from active metadata without opening any connector."""
    active = sources.list_active_sources()
    if connection_id is None:
        if len(active) != 1:
            raise ValueError("source_required: select an active connection_id")
        return active[0]
    for source in active:
        if source.connection_id == connection_id:
            return source
    raise ValueError("invalid_source_selector: unknown or inactive connection_id")


def validate_custom_analysis_sql(
    sql: str, *, connection_id: str | None = None, analysis_id: str | None = None
) -> dict[str, Any]:
    """Validate SQL in one active source, inheriting existing ownership.

    Does not persist anything — used by the "Validate SQL" step before a
    CustomAnalysis is created or updated.

    Raises :class:`CustomAnalysisSqlError` when the SQL can't be resolved.
    """
    if analysis_id is not None:
        if get_custom_analysis_by_id(analysis_id) is None:
            raise ValueError(f"CustomAnalysis {analysis_id!r} not found")
        connection_id = _definition_connection_id(analysis_id, connection_id)
    source = _resolve_source(connection_id)
    validate_sql(
        sql,
        get_dialects(source.connection_id),
        get_schemas(source.connection_id),
        connection_id=source.connection_id,
    )
    return {"valid": True, "sql": sql}


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _embed_analysis(analysis_id: str) -> None:
    """Embed an analysis scoped to the database whose tables it references.

    Semantic search filters hits on ``database_name`` (see
    :func:`auto_ontology.retrieval.data_access.semantic_search.search_semantic_index`).
    """
    from auto_ontology.utils import get_embed_params
    from auto_ontology.vdb import get_semantic_vdb

    from auto_ontology.dal.custom_analyses import get_custom_analysis_connection_id

    connection_id = get_custom_analysis_connection_id(analysis_id)
    if connection_id is None:
        logger.info("Skipping unowned CustomAnalysis %s embedding", analysis_id)
        return
    vdb = get_semantic_vdb(connection_id=connection_id)
    database_name = fetch_database_name_for_analysis(analysis_id)
    if database_name is None:
        return

    embed_custom_analyses(
        embed_params=get_embed_params(),
        vdb=vdb,
        analysis_id=analysis_id,
        database_name=database_name,
        connection_id=connection_id,
    )


def _persist_analysis_with_sql(
    analysis_node: CatalogNode,
    sql: str,
    query_obj: Any,
) -> dict[str, Any]:
    """Link a pre-parsed ``query_obj`` to ``analysis_node`` via HAS_SQL."""
    query_obj.sql_node.match_props = {
        "sql_full_query": sql,
        "connection_id": query_obj.connection_id,
    }

    edge_props = {Props.ANALYSIS_ID: analysis_node.get_id()}
    query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

    add_query(query_obj.get_edges(), connection_id=query_obj.connection_id)

    props = analysis_node.get_properties()
    return {
        "id": analysis_node.get_id(),
        "name": props["name"],
        "description": props["description"],
        "sql": sql,
        "connection_id": query_obj.connection_id,
    }


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def create_custom_analysis(
    name: str,
    description: str,
    sql: str,
    *,
    embed: bool = True,
    connection_id: str | None = None,
) -> dict[str, Any]:
    """Create a fresh ``CustomAnalysis`` linked to its ``Sql`` node.

    Raises :class:`CustomAnalysisNameConflict` when ``name`` is already used.
    Raises :class:`CustomAnalysisSqlConflict` when ``sql`` is already attached.
    Raises :class:`CustomAnalysisSqlError` when the SQL can't be resolved.
    Select ``connection_id`` when multiple sources are active; conflict checks
    and SQL dependencies use only that source. The returned metadata keeps
    ``connection_id`` separate from the physical ``database_name``.

    *embed* exists for bulk loaders, and defaults to the interactive
    behaviour. Embedding inline is right when a user creates one analysis --
    it has to be searchable when the request returns. It is wrong when a
    seeder creates nineteen: each embed is its own network round trip, so the
    load costs one latency per analysis (~2s each, ~40s for nineteen)
    regardless of how little work the database does.

    Passing ``embed=False`` makes the caller responsible for embedding, via
    :func:`auto_ontology.dal.custom_analyses.embed_custom_analyses` over the whole set.
    A caller that batches must pass it: embedding here *and* batching
    afterwards writes every analysis into the semantic index twice, and the
    index has no uniqueness constraint to catch it -- retrieval simply starts
    returning the same analysis in two of its top-k slots.
    """
    source = _resolve_source(connection_id)
    connection_id = source.connection_id
    name_conflict = find_analysis_by_name(
        name, exclude_id=None, connection_id=connection_id
    )
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = find_analysis_by_sql(
        sql, exclude_id=None, connection_id=connection_id
    )
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    query_obj = validate_sql(
        sql,
        get_dialects(connection_id),
        get_schemas(connection_id),
        connection_id=connection_id,
    )
    analysis_node = CatalogNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={
            "name": name,
            "description": description,
            "connection_id": connection_id,
        },
        match_props={"name": name, "connection_id": connection_id},
    )

    row = _persist_analysis_with_sql(analysis_node, sql, query_obj)
    row["database_name"] = source.database_name

    if embed:
        _embed_analysis(row["id"])

    return row


def _definition_connection_id(analysis_id: str, connection_id: str | None) -> str:
    stored_id = analysis_dal.get_custom_analysis_connection_id(analysis_id)
    if connection_id is not None and connection_id != stored_id:
        raise ValueError("source_conflict: definition ownership is immutable")
    if stored_id is None:
        raise ValueError("source_required: legacy definition has no source ownership")
    return stored_id


def update_custom_analysis(
    analysis_id: str,
    name: str,
    description: str,
    sql: str,
    *,
    connection_id: str | None = None,
) -> dict[str, Any] | None:
    """Replace name/description/sql of an existing ``CustomAnalysis`` by id.

    Inherit stored source ownership; a conflicting explicit ID or unresolved
    legacy definition is rejected before SQL validation or metadata writes.
    Returns the updated row or ``None`` when no analysis with ``analysis_id`` exists.
    Raises :class:`CustomAnalysisNameConflict`, :class:`CustomAnalysisSqlConflict`,
    or :class:`CustomAnalysisSqlError` on constraint violations.
    """
    if get_custom_analysis_by_id(analysis_id) is None:
        return None

    connection_id = _definition_connection_id(analysis_id, connection_id)
    source = _resolve_source(connection_id)
    name_conflict = find_analysis_by_name(
        name, exclude_id=analysis_id, connection_id=connection_id
    )
    if name_conflict is not None:
        raise CustomAnalysisNameConflict(
            f"another CustomAnalysis already uses name {name!r} "
            f"(id={name_conflict['id']!r})",
        )

    sql_conflict = find_analysis_by_sql(
        sql, exclude_id=analysis_id, connection_id=connection_id
    )
    if sql_conflict is not None:
        raise CustomAnalysisSqlConflict(
            f"this SQL is already used by CustomAnalysis {sql_conflict['name']!r} "
            f"(id={sql_conflict['id']!r})",
        )

    # Validate BEFORE touching the graph so a parse failure can't orphan
    # the analysis from its Sql node.
    query_obj = validate_sql(
        sql,
        get_dialects(connection_id),
        get_schemas(connection_id),
        connection_id=connection_id,
    )

    analysis_node = CatalogNode(
        name=name,
        label=Labels.CUSTOM_ANALYSIS,
        props={
            "name": name,
            "description": description,
            "connection_id": connection_id,
        },
        match_props={"id": analysis_id, "connection_id": connection_id},
        existing_id=analysis_id,
        override_existing_props={"name": name, "description": description},
    )

    with write_transaction():
        detach_existing_sql_edges(analysis_id)
        row = _persist_analysis_with_sql(analysis_node, sql, query_obj)
    row["database_name"] = source.database_name

    _embed_analysis(analysis_id)

    return row


def delete_custom_analysis(analysis_id: str) -> dict[str, str] | None:
    """Remove a CustomAnalysis, its Sql node, and its VDB embedding.

    Returns ``{"id": analysis_id}`` on success, or ``None`` when not found.
    """
    if get_custom_analysis_by_id(analysis_id) is None:
        return None

    from auto_ontology.dal.custom_analyses import get_custom_analysis_connection_id

    connection_id = get_custom_analysis_connection_id(analysis_id)
    delete_custom_analysis_node(analysis_id)

    from auto_ontology.vdb import get_semantic_vdb

    if connection_id:
        get_semantic_vdb(connection_id=connection_id).delete_by_id(
            analysis_id, connection_id=connection_id
        )

    return {"id": analysis_id}
