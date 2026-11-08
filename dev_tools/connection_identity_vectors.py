# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vector migration receipts; never infer dimensions or reuse mixed semantic text."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg import sql

from auto_ontology.vdb import DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION, VDB_SCHEMA
from auto_ontology.vdb.records import projection_id


@dataclass(frozen=True)
class VectorUpdate:
    previous_id: str
    projection: str
    connection_id: str | None
    metadata: dict[str, Any] = field(repr=False)


@dataclass
class CollectionPlan:
    name: str
    exists: bool = False
    has_source_column: bool = False
    has_source_index: bool = False
    updates: list[VectorUpdate] = field(default_factory=list)
    unresolved_ids: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    current: dict[str, set[str]] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "source_schema_ready": self.exists
            and self.has_source_column
            and self.has_source_index,
            "pending_updates": len(self.updates),
            "unresolved_ids": self.unresolved_ids,
            "conflict_ids": self.conflicts,
        }


@dataclass
class VectorPlan:
    collections: dict[str, CollectionPlan]
    expected: dict[str, dict[str, set[str]]]
    sources: dict[str, str]
    active_ids: set[str]

    def missing_semantic(self) -> list[str]:
        semantic = self.collections[SEMANTIC_VDB_COLLECTION]
        return [
            owner
            for owner, expected in self.expected[SEMANTIC_VDB_COLLECTION].items()
            if expected - semantic.current.get(owner, set())
        ]

    def report(self) -> dict[str, Any]:
        ready = {}
        for owner in sorted(self.active_ids):
            data_expected = self.expected[DATA_VDB_COLLECTION].get(owner, set())
            ready[owner] = bool(data_expected) and all(
                expected.get(owner, set())
                <= self.collections[collection].current.get(owner, set())
                for collection, expected in self.expected.items()
            )
        complete = not self.missing_semantic() and all(
            not table.conflicts
            and not table.updates
            and (
                not table.exists or (table.has_source_column and table.has_source_index)
            )
            for table in self.collections.values()
        )
        return {
            "vectors": {
                name: table.report() for name, table in self.collections.items()
            },
            "semantic_rebuild_required": self.missing_semantic(),
            "vector_migration_complete": complete,
            "sources_ready_for_querying": ready,
            "ready_for_querying": complete and bool(ready) and all(ready.values()),
        }


def _expected_semantic_ids(connection_id: str) -> set[str]:
    from auto_ontology.dal.terms import fetch_all_terms_and_attributes
    from auto_ontology.dal.sql_attributes import list_sql_attributes
    from auto_ontology.dal.custom_analyses import list_custom_analyses

    terms, attrs = fetch_all_terms_and_attributes(connection_id=connection_id)
    definitions = list_sql_attributes(
        connection_id=connection_id
    ) + list_custom_analyses(connection_id=connection_id)
    return {str(row["id"]) for row in terms + attrs + definitions}


def _table_columns(conn: psycopg.Connection, name: str) -> set[str]:
    return {
        row["column_name"]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",
            (VDB_SCHEMA, name),
        )
    }


def _updated_metadata(
    metadata: dict[str, Any], connection_id: str | None
) -> dict[str, Any]:
    return {
        **metadata,
        "connection_id": connection_id,
        "content_metadata": {
            **(metadata.get("content_metadata") or {}),
            "connection_id": connection_id,
        },
    }


def _read_collection(
    conn: psycopg.Connection, name: str
) -> tuple[CollectionPlan, list[dict[str, Any]]]:
    columns = _table_columns(conn, name)
    plan = CollectionPlan(
        name, exists=bool(columns), has_source_column="connection_id" in columns
    )
    if not columns:
        return plan, []
    if not {"langchain_id", "langchain_metadata", "embedding", "content"} <= columns:
        plan.conflicts.append("unsupported_vector_schema")
        return plan, []
    if plan.has_source_column:
        column = conn.execute(
            "SELECT data_type,is_nullable FROM information_schema.columns WHERE table_schema=%s AND table_name=%s AND column_name='connection_id'",
            (VDB_SCHEMA, name),
        ).fetchone()
        if column != {"data_type": "text", "is_nullable": "YES"}:
            plan.conflicts.append("unsupported_source_column")
            return plan, []
    plan.has_source_index = bool(
        conn.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND tablename=%s AND indexname=%s",
            (VDB_SCHEMA, name, f"ix_{name}_connection_id"),
        ).fetchone()
    )
    promoted = [
        sql.Identifier(key) if key in columns else sql.SQL("NULL::text")
        for key in ("connection_id", "label", "database_name")
    ]
    rows = conn.execute(
        sql.SQL(
            "SELECT langchain_id::text AS vector_id,langchain_metadata AS metadata,{} AS connection_id,{} AS label,{} AS database_name FROM {} ORDER BY langchain_id"
        ).format(*promoted, sql.Identifier(VDB_SCHEMA, name))
    ).fetchall()
    valid = []
    for row in rows:
        metadata = {} if row["metadata"] is None else row["metadata"]
        if not _valid_metadata(metadata):
            plan.conflicts.append(row["vector_id"])
            continue
        nested = metadata.get("content_metadata") or {}
        if any(
            len(
                {
                    obj.get(key)
                    for obj in (row, metadata, nested)
                    if obj.get(key) is not None
                    and (key != "database_name" or obj[key] != "")
                }
            )
            > 1
            for key in ("id", "connection_id", "label", "database_name")
        ):
            plan.conflicts.append(row["vector_id"])
            continue
        valid.append(
            {
                **row,
                "metadata": metadata,
                "label": row["label"] or metadata.get("label") or nested.get("label"),
                "database_name": row["database_name"]
                or metadata.get("database_name")
                or nested.get("database_name"),
            }
        )
    return plan, valid


def _valid_metadata(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    nested = value.get("content_metadata") or {}
    if not isinstance(nested, dict):
        return False
    return all(
        obj.get(key) is None or isinstance(obj[key], str)
        for obj in (value, nested)
        for key in ("id", "connection_id", "database_name", "label")
    )


def _schedule(
    plan: CollectionPlan, row: dict[str, Any], target: str, owner: str | None
) -> None:
    metadata = _updated_metadata(row["metadata"] or {}, owner)
    # PGVectorStore promotes CID out of JSON on every write.
    stored_metadata = row["metadata"] or {}
    if plan.has_source_column:
        stored_metadata = {"connection_id": row["connection_id"], **stored_metadata}
    if (
        not plan.has_source_column
        or row["vector_id"] != target
        or row["connection_id"] != owner
        or stored_metadata != metadata
    ):
        plan.updates.append(VectorUpdate(row["vector_id"], target, owner, metadata))
    if (
        owner
        and plan.has_source_column
        and row["connection_id"] == owner
        and row["vector_id"] == target
        and stored_metadata == metadata
    ):
        plan.current.setdefault(owner, set()).add(str(metadata["id"]))


def _data_collection(
    plan: CollectionPlan,
    rows: list[dict],
    owners: dict[str, str],
    names: dict[str, str],
) -> None:
    for row in rows:
        meta = row["metadata"] or {}
        node_id = meta.get("id") or (meta.get("content_metadata") or {}).get("id")
        owner = owners.get(node_id)
        if owner is None:
            plan.unresolved_ids.append(row["vector_id"])
            _schedule(plan, row, row["vector_id"], None)
            continue
        if (
            row["connection_id"] not in (None, owner)
            or meta.get("connection_id") not in (None, owner)
            or (meta.get("content_metadata") or {}).get("connection_id")
            not in (None, owner)
            or row["database_name"] not in (None, "", names[owner])
        ):
            plan.conflicts.append(row["vector_id"])
            continue
        target = projection_id(plan.name, owner, node_id)
        if not meta.get("id"):
            row = {**row, "metadata": {**meta, "id": node_id}}
        _schedule(plan, row, target, owner)


def _semantic_collection(
    plan: CollectionPlan,
    rows: list[dict],
    expected: dict[str, set[str]],
    pql_ids: set[str],
) -> None:
    for row in rows:
        meta = row["metadata"] or {}
        node_id = meta.get("id") or (meta.get("content_metadata") or {}).get("id")
        if row["label"] == "PqlAnalysis" and node_id in pql_ids:
            from auto_ontology.vdb.records import global_pql_projection_id

            if any(
                obj.get("connection_id") is not None
                for obj in (row, meta, meta.get("content_metadata") or {})
            ):
                plan.conflicts.append(row["vector_id"])
                continue
            _schedule(plan, row, global_pql_projection_id(plan.name, node_id), None)
            continue
        owner = row["connection_id"] or meta.get("connection_id")
        if (
            owner
            and node_id in expected.get(owner, set())
            and row["vector_id"] == projection_id(plan.name, owner, node_id)
        ):
            _schedule(plan, row, row["vector_id"], owner)
        else:
            # Random-ID legacy semantic text may already contain mixed-source samples.
            plan.unresolved_ids.append(row["vector_id"])
            _schedule(plan, row, row["vector_id"], None)


def _check_target_collisions(plan: CollectionPlan, rows: list[dict]) -> None:
    targets = {update.previous_id: update.projection for update in plan.updates}
    final_ids = {
        row["vector_id"]: targets.get(row["vector_id"], row["vector_id"])
        for row in rows
    }
    counts = Counter(final_ids.values())
    plan.conflicts = sorted(
        set(plan.conflicts)
        | {previous for previous, target in final_ids.items() if counts[target] > 1}
    )


def inspect_vectors(conn: psycopg.Connection) -> VectorPlan:
    """Read-only plan; never read vector contents, credentials or embedding models."""
    roots = conn.execute(
        "SELECT id,name,configuration_origin FROM catalog_database ORDER BY id"
    ).fetchall()
    names = {row["id"]: row["name"] for row in roots}
    rows = conn.execute(
        "SELECT t.id,s.database_id FROM catalog_table t JOIN catalog_schema s ON s.id=t.schema_id UNION ALL "
        "SELECT c.id,s.database_id FROM catalog_column c JOIN catalog_table t ON t.id=c.table_id JOIN catalog_schema s ON s.id=t.schema_id"
    ).fetchall()
    owner_sets: dict[str, set[str]] = {}
    for row in rows:
        owner_sets.setdefault(row["id"], set()).add(row["database_id"])
    owners = {
        node: next(iter(ids)) for node, ids in owner_sets.items() if len(ids) == 1
    }
    data_expected = {
        owner: {node for node, source in owners.items() if source == owner}
        for owner in names
    }
    semantic_expected = {owner: _expected_semantic_ids(owner) for owner in names}
    data, data_rows = _read_collection(conn, DATA_VDB_COLLECTION)
    _data_collection(data, data_rows, owners, names)
    semantic, semantic_rows = _read_collection(conn, SEMANTIC_VDB_COLLECTION)
    pql_ids = {row["id"] for row in conn.execute("SELECT id FROM pql_analysis")}
    _semantic_collection(semantic, semantic_rows, semantic_expected, pql_ids)
    _check_target_collisions(data, data_rows)
    _check_target_collisions(semantic, semantic_rows)
    return VectorPlan(
        {data.name: data, semantic.name: semantic},
        {data.name: data_expected, semantic.name: semantic_expected},
        names,
        {row["id"] for row in roots if row["configuration_origin"] is not None},
    )


def apply_vectors(conn: psycopg.Connection, plan: VectorPlan) -> None:
    """Backfill identities without reading/recomputing embeddings or creating tables."""
    for table in plan.collections.values():
        if not table.exists:
            continue
        if table.conflicts:
            raise ValueError("vector_identity_conflict")
        qualified = sql.Identifier(VDB_SCHEMA, table.name)
        if not table.has_source_column:
            conn.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN connection_id TEXT").format(
                    qualified
                )
            )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (connection_id)").format(
                sql.Identifier(f"ix_{table.name}_connection_id"), qualified
            )
        )
        for row in table.updates:
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET langchain_id=%s::uuid,connection_id=%s,langchain_metadata=%s::jsonb WHERE langchain_id=%s::uuid"
                ).format(qualified),
                (
                    row.projection,
                    row.connection_id,
                    json.dumps(row.metadata),
                    row.previous_id,
                ),
            )


def rebuild_semantic(connection_id: str, database_name: str) -> bool:
    """Rebuild from scoped metadata; an unavailable model leaves a retryable receipt."""
    from auto_ontology.semantic.embed import (
        build_semantic_embedder,
        embed_all_semantic_nodes,
    )
    from auto_ontology.dal import custom_analyses, sql_attributes
    from auto_ontology.utils.embedding import embed_docs_into_vdb

    embedder = build_semantic_embedder(
        database_name, connection_id=connection_id, reset=False
    )
    if embedder is None:
        return False
    try:
        embed_all_semantic_nodes(embedder)
        for attribute in sql_attributes.list_sql_attributes(
            connection_id=connection_id
        ):
            docs = sql_attributes.fetch_sql_attribute_docs(
                attribute["id"], connection_id=connection_id
            )
            embed_docs_into_vdb(
                docs,
                embedder.embed_params,
                embedder.vdb,
                database_name,
                connection_id=connection_id,
            )
        custom_analyses.embed_custom_analyses(
            embedder.embed_params,
            embedder.vdb,
            database_name=database_name,
            connection_id=connection_id,
        )
    finally:
        embedder.vdb.close()
    return True
