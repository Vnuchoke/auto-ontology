# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SqlAttribute write orchestration.

All direct store access lives in auto_ontology/dal/sql_attributes.py.
This module keeps orchestration: connector resolution, SQL validation,
node persistence via add_query, and VDB embedding lifecycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from auto_ontology.catalog.store.queries import add_query
from auto_ontology.catalog.model import CatalogNode
from auto_ontology.catalog.constants import Props

from auto_ontology.connectors import sources
from auto_ontology.dal import sql_attributes as sql_attr_dal
from auto_ontology.dal.session import write_transaction
from auto_ontology.dal.pii import realign_sql_attribute_pii
from auto_ontology.dal.sql_attributes import (
    SqlAttributeExpressionConflict,
    SqlAttributeNameConflict,
    SqlAttributeSqlError,
    clear_sql_attribute_description_suggestion,
    delete_sql_attribute_node,
    detach_existing_sql_edges,
    fetch_sql_attribute_docs,
    find_attr_by_expression,
    find_attr_by_name,
    get_full_sql_attribute_by_id,
    get_sql_attribute_by_id,
    link_to_term,
    list_sql_attributes,
)
from auto_ontology.dal.tags import PII_TAG_NAME, get_tag_by_name
from auto_ontology.dal.terms import get_slim_term_by_id
from auto_ontology.semantic.constants import LABEL_SQL_ATTRIBUTE
from auto_ontology.server.sql_attributes.description_suggester import (
    suggest_sql_attribute_description,
)
from auto_ontology.server.sql_utils import get_dialects, get_schemas, validate_sql
from auto_ontology.utils import get_embed_params
from auto_ontology.utils.embedding import embed_docs_into_vdb
from auto_ontology.vdb import get_semantic_vdb

if TYPE_CHECKING:
    from auto_ontology.utils.embedding import EmbedParams
    from auto_ontology.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

__all__ = [
    "SqlAttributeNameConflict",
    "SqlAttributeExpressionConflict",
    "SqlAttributeSqlError",
    "list_sql_attributes",
    "get_full_sql_attribute_by_id",
    "get_sql_attribute_by_id",
    "validate_sql_attribute",
    "create_sql_attribute",
    "update_sql_attribute",
    "delete_sql_attribute",
    "suggest_sql_attribute_description",
]


# ---------------------------------------------------------------------------
# Write helpers (private)
# ---------------------------------------------------------------------------


def _resolve_source(
    connection_id: str | None, connector: str | None = None
) -> sources.SourceDescriptor:
    """Resolve metadata only; legacy physical names must be unambiguous."""
    active = sources.list_active_sources()
    if connector is not None:
        matches = [
            s for s in active if s.database_name.casefold() == connector.casefold()
        ]
        if not connector.strip() or len(matches) != 1:
            raise ValueError("invalid_source_selector: ambiguous or unknown connector")
        if connection_id is not None and connection_id != matches[0].connection_id:
            raise ValueError("source_conflict: connector and connection_id differ")
        connection_id = matches[0].connection_id
    if connection_id is None:
        if len(active) != 1:
            raise ValueError("source_required: select an active connection_id")
        return active[0]
    for source in active:
        if source.connection_id == connection_id:
            return source
    raise ValueError("invalid_source_selector: unknown or inactive connection_id")


def _definition_source(
    existing: dict[str, Any], connection_id: str | None, connector: str | None
) -> sources.SourceDescriptor:
    stored_id = existing.get("connection_id")
    if connection_id is not None and connection_id != stored_id:
        raise ValueError("source_conflict: definition ownership is immutable")
    if stored_id is None:
        raise ValueError("source_required: legacy definition has no source ownership")
    return _resolve_source(stored_id, connector)


def _persist_attr_with_sql(
    attr_node: CatalogNode,
    sql: str,
    query_obj: Any,
) -> dict[str, Any]:
    """Create the Sql node via add_query and link it to the SqlAttribute."""
    query_obj.sql_node.match_props = {
        "sql_full_query": sql,
        "connection_id": query_obj.connection_id,
    }

    edge_props = {Props.ANALYSIS_ID: attr_node.get_id()}
    query_obj.edges.append((attr_node, query_obj.sql_node, edge_props))

    add_query(query_obj.get_edges(), connection_id=query_obj.connection_id)

    props = attr_node.get_properties()
    return {
        "id": attr_node.get_id(),
        "name": props["name"],
        "description": props.get("description", ""),
        "expression": props.get("expression", ""),
        "source": props.get("source", ""),
        "sql_id": query_obj.sql_node.get_id(),
        "sql": sql,
        "certified": props.get("certified", False),
        "owner_id": props.get("owner_id"),
        "connection_id": query_obj.connection_id,
    }


def _embed_sql_attribute(
    embed_params: "EmbedParams",
    vdb: "PostgresVDB",
    attr_id: str,
    database_name: str | None = None,
    *,
    connection_id: str,
) -> None:
    """Fetch docs from the store, embed them, and upsert into *vdb*."""
    docs = fetch_sql_attribute_docs(attr_id, connection_id=connection_id)
    if not docs:
        logger.info(
            "No SqlAttribute rows found for attr_id=%r; skipping VDB upsert.",
            attr_id,
        )
        return
    embed_docs_into_vdb(
        docs, embed_params, vdb, database_name, connection_id=connection_id
    )


# ---------------------------------------------------------------------------
# Validation API
# ---------------------------------------------------------------------------


def validate_sql_attribute(
    *,
    expression: str,
    term_id: str | None = None,
    attribute_id: str | None = None,
    connector: str | None = None,
    connection_id: str | None = None,
) -> dict[str, Any]:
    """Validate a SQL expression against the catalog.

    Does not persist anything — used by the "Validate SQL" step before a
    SqlAttribute is created or updated. Select ``connection_id`` when multiple
    sources are active; existing attributes inherit immutable ownership.
    The legacy internal ``connector`` name must resolve unambiguously.

    When *term_id* is given, also checks that no other SqlAttribute of the
    same term already uses an equivalent expression (*attribute_id*, when
    editing an existing attribute, excludes it from that check).

    Raises :class:`SqlAttributeExpressionConflict` on a duplicate SQL
    snippet. Raises :class:`SqlAttributeSqlError` when the SQL can't be
    resolved.
    """
    if attribute_id is not None:
        existing = get_full_sql_attribute_by_id(attribute_id)
        if existing is None:
            raise ValueError(f"SqlAttribute {attribute_id!r} not found")
        source = _definition_source(existing, connection_id, connector)
    else:
        source = _resolve_source(connection_id, connector)
    connection_id = source.connection_id
    if term_id is not None:
        conflict = find_attr_by_expression(
            term_id=term_id,
            expression=expression,
            exclude_id=attribute_id,
            connection_id=connection_id,
        )
        if conflict is not None:
            raise SqlAttributeExpressionConflict("SQL snippet already exists")

    validate_sql(
        expression,
        get_dialects(connection_id),
        get_schemas(connection_id),
        connection_id=connection_id,
    )
    return {"valid": True, "expression": expression}


def _reembed_sql_attribute(
    attr_id: str, database_name: str | None, *, connection_id: str
) -> None:
    vdb = get_semantic_vdb(connection_id=connection_id)
    docs = fetch_sql_attribute_docs(attr_id, connection_id=connection_id)
    # Build and embed before removing the previous projection.
    from auto_ontology.utils.embedding import embed_rows
    from auto_ontology.utils.embedding_rows import build_embed_row
    from auto_ontology.vdb.records import write_embedded_rows

    rows = [
        build_embed_row(
            text=doc["text"],
            node_id=doc["id"],
            label=doc["label"],
            name=doc["name"],
            schema_name="",
            database_name=database_name,
            connection_id=connection_id,
        )
        for doc in docs
    ]
    for row, doc in zip(rows, docs):
        row["metadata"]["source"] = doc.get("source", "")
        row["metadata"]["content_metadata"]["source"] = doc.get("source", "")
    embedded = embed_rows(rows, get_embed_params(), what="SqlAttribute")
    vdb.delete_by_id(attr_id, connection_id=connection_id)
    write_embedded_rows(vdb, embedded)


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------


def create_sql_attribute(
    *,
    name: str,
    description: str,
    expression: str,
    term_id: str,
    connector: str | None = None,
    connection_id: str | None = None,
    source: str = "manual",
    owner_id: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    """Create a SqlAttribute, its Sql node, link to a Term, and embed.

    Select ``connection_id`` with multiple active sources. The legacy internal
    ``connector`` physical name is accepted only when unambiguous.
    ``owner_id`` is stored when given; ``None`` leaves the attribute
    unassigned. The caller has already checked that the id may own it.
    ``created_by`` is the user id from the gateway. Compilation leaves it
    unset, which the panel shows as Auto Generated.

    Raises :class:`SqlAttributeNameConflict` when ``name`` is already used.
    Raises :class:`SqlAttributeSqlError` when the SQL can't be resolved.
    Raises ``ValueError`` when the Term doesn't exist.
    """
    resolved = _resolve_source(connection_id, connector)
    connection_id = resolved.connection_id
    database_name = resolved.database_name

    conflict = find_attr_by_name(name, exclude_id=None, connection_id=connection_id)
    if conflict is not None:
        raise SqlAttributeNameConflict(
            f"SqlAttribute with name {name!r} already exists (id={conflict['id']!r})"
        )

    term = get_slim_term_by_id(term_id)
    if not term:
        raise ValueError(f"Term with id {term_id!r} not found")

    expression_conflict = find_attr_by_expression(
        term_id=term_id,
        expression=expression,
        exclude_id=None,
        connection_id=connection_id,
    )
    if expression_conflict is not None:
        raise SqlAttributeExpressionConflict("SQL snippet already exists")

    query_obj = validate_sql(
        expression,
        get_dialects(connection_id),
        get_schemas(connection_id),
        connection_id=connection_id,
    )

    props = {
        "name": name,
        "description": description,
        "expression": expression,
        "source": source,
        "connection_id": connection_id,
    }
    if owner_id is not None:
        props["owner_id"] = owner_id
    if created_by is not None:
        props["created_by"] = created_by
    attr_node = CatalogNode(
        name=name,
        label=LABEL_SQL_ATTRIBUTE,
        props=props,
        match_props={"name": name, "connection_id": connection_id},
    )

    with write_transaction():
        row = _persist_attr_with_sql(attr_node, expression, query_obj)
        link_to_term(row["id"], term_id)
    row["term_name"] = term["name"]
    row["term_id"] = term_id
    row["database_name"] = database_name

    vdb = get_semantic_vdb(connection_id=connection_id)
    _embed_sql_attribute(
        embed_params=get_embed_params(),
        vdb=vdb,
        attr_id=row["id"],
        database_name=database_name,
        connection_id=connection_id,
    )

    return row


def update_sql_attribute(
    *,
    attr_id: str,
    name: str | None = None,
    description: str | None = None,
    expression: str | None = None,
    term_id: str | None = None,
    connector: str | None = None,
    connection_id: str | None = None,
    source: str = "manual",
    certified: bool | None = None,
    owner_id: str | None = None,
    set_owner: bool = False,
    modified_by: str | None = None,
    touch: bool = False,
) -> dict[str, Any] | None:
    """Update a SqlAttribute.

    When *expression* is given, replaces the full record: re-parses SQL,
    re-links the Term, and re-embeds. *name* and *term_id* are required in
    that case.

    When *expression* is omitted, patches name/description and/or the
    certification flag only. The certification flag carries no embedding
    content, so a certification-only patch skips the VDB refresh.

    SQL replacement inherits stored source ownership; explicit selectors must
    agree with it. Unresolved legacy records cannot be rebound by an edit.

    Returns the updated row or ``None`` when no SqlAttribute with
    ``attr_id`` exists.
    """
    existing = get_full_sql_attribute_by_id(attr_id)
    if existing is None:
        return None

    if connection_id is not None and connection_id != existing.get("connection_id"):
        raise ValueError("source_conflict: definition ownership is immutable")
    connection_id = existing.get("connection_id")

    if expression is not None:
        if name is None or term_id is None:
            raise ValueError(
                "name and term_id are required when updating SQL expression"
            )

        next_name = name.strip()
        next_description = (
            description if description is not None else existing.get("description", "")
        )
        resolved = _definition_source(existing, connection_id, connector)
        database_name = resolved.database_name

        name_changed = (existing.get("name") or "").strip() != next_name
        expression_changed = (
            existing.get("expression") or ""
        ).strip() != expression.strip()

        conflict = find_attr_by_name(
            next_name, exclude_id=attr_id, connection_id=connection_id
        )
        if conflict is not None:
            raise SqlAttributeNameConflict(
                f"SqlAttribute with name {next_name!r} already exists "
                f"(id={conflict['id']!r})"
            )

        term = get_slim_term_by_id(term_id)
        if not term:
            raise ValueError(f"Term with id {term_id!r} not found")

        expression_conflict = find_attr_by_expression(
            term_id=term_id,
            expression=expression,
            exclude_id=attr_id,
            connection_id=connection_id,
        )
        if expression_conflict is not None:
            raise SqlAttributeExpressionConflict("SQL snippet already exists")

        query_obj = validate_sql(
            expression,
            get_dialects(connection_id),
            get_schemas(connection_id),
            connection_id=connection_id,
        )

        attr_node = CatalogNode(
            name=next_name,
            label=LABEL_SQL_ATTRIBUTE,
            props={
                "name": next_name,
                "description": next_description,
                "expression": expression,
                "source": source,
                "connection_id": connection_id,
            },
            match_props={"id": attr_id, "connection_id": connection_id},
            existing_id=attr_id,
        )

        with write_transaction():
            detach_existing_sql_edges(attr_id)
            if name_changed or expression_changed:
                clear_sql_attribute_description_suggestion(attr_id)
            sql_attr_dal.update_sql_attribute(
                attr_id,
                name=next_name,
                description=next_description,
                expression=expression,
                source=source,
                modified_by=modified_by,
                touch=touch,
            )
            _persist_attr_with_sql(attr_node, expression, query_obj)
            link_to_term(attr_id, term_id)
            if expression_changed:
                pii_tag = get_tag_by_name(PII_TAG_NAME)
                realign_sql_attribute_pii(
                    attr_id,
                    str(pii_tag["id"]) if pii_tag else None,
                )
        _reembed_sql_attribute(attr_id, database_name, connection_id=connection_id)

        return {
            "id": attr_id,
            "name": next_name,
            "description": next_description,
            "expression": expression,
            "source": source,
            "database_name": database_name,
            "connection_id": connection_id,
            "term_name": term["name"],
            "term_id": term_id,
            "sql": expression,
            "certified": existing.get("certified", False),
            "owner_id": existing.get("owner_id"),
        }

    next_name = name.strip() if name is not None else existing.get("name", "")
    if not next_name:
        raise ValueError("SQL attribute name cannot be blank")

    next_description = (
        description if description is not None else existing.get("description", "")
    )

    name_changed = (existing.get("name") or "").strip() != next_name
    description_changed = (existing.get("description") or "") != next_description
    certification_changed = certified is not None
    owner_changed = set_owner and existing.get("owner_id") != owner_id
    if (
        not name_changed
        and not description_changed
        and not certification_changed
        and not owner_changed
    ):
        return existing

    if name_changed:
        conflict = find_attr_by_name(
            next_name, exclude_id=attr_id, connection_id=connection_id
        )
        if conflict is not None:
            raise SqlAttributeNameConflict(
                f"SqlAttribute with name {next_name!r} already exists "
                f"(id={conflict['id']!r})"
            )
        clear_sql_attribute_description_suggestion(attr_id)

    sql_attr_dal.update_sql_attribute(
        attr_id,
        name=next_name if name_changed else None,
        description=next_description if description_changed else None,
        certified=certified,
        owner_id=owner_id,
        set_owner=set_owner,
        modified_by=modified_by,
        touch=touch,
    )
    if name_changed or description_changed:
        if connection_id is not None:
            _reembed_sql_attribute(
                attr_id, existing.get("database_name"), connection_id=connection_id
            )

    return get_full_sql_attribute_by_id(attr_id)


def delete_sql_attribute(attr_id: str) -> dict[str, Any] | None:
    """Delete a SqlAttribute, its edges, and its VDB embedding.

    Returns ``{"id": attr_id}`` on success, or ``None`` when not found.
    """
    existing = get_full_sql_attribute_by_id(attr_id)
    if existing is None:
        return None

    delete_sql_attribute_node(attr_id)
    if existing.get("connection_id"):
        get_semantic_vdb(connection_id=existing["connection_id"]).delete_by_id(
            attr_id, connection_id=existing["connection_id"]
        )

    return {"id": attr_id}
