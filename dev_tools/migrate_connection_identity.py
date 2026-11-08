# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect/backfill connection metadata after the additive Alembic revision.

Does not load .env, read Vault, or open an analyzed database. POSTGRES_* must
explicitly select the authorized store. Semantic rebuild uses the configured
embedding service only after metadata commits, and reports incomplete work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from auto_ontology.infra.postgres import get_postgres_connection_string
from dev_tools.connection_identity_vectors import (
    apply_vectors,
    inspect_vectors,
    rebuild_semantic,
)


class Origin(StrEnum):
    MANAGED = "managed"
    ENV = "env"


@dataclass(frozen=True)
class IdentityOverride:
    display_label: str | None = None
    configuration_origin: Origin | None = None
    credential_ref: str | None = None
    origin_specified: bool = False


def _nonblank(value: object) -> str:
    if not isinstance(value, str) or not value.strip(" "):
        raise ValueError("Identity mapping values must be nonblank strings")
    return value.strip(" ")


def _override(value: object) -> IdentityOverride:
    allowed = {"display_label", "configuration_origin", "credential_ref"}
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(
            "Identity mapping accepts only label, origin and credential ref"
        )
    label = _nonblank(value["display_label"]) if "display_label" in value else None
    reference = value.get("credential_ref")
    if reference is not None:
        _nonblank(reference)
    origin = value.get("configuration_origin")
    return IdentityOverride(
        display_label=label,
        configuration_origin=Origin(origin) if origin is not None else None,
        credential_ref=reference,
        origin_specified="configuration_origin" in value,
    )


def validate_mapping(value: object) -> dict[str, IdentityOverride]:
    if not isinstance(value, dict):
        raise ValueError("Identity mapping must be an object keyed by existing IDs")
    result = {}
    for key, item in value.items():
        _nonblank(key)
        result[key] = _override(item)
    return result


@dataclass(frozen=True)
class Issue:
    code: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class RootUpdate:
    id: str
    display_label: str
    configuration_origin: Origin | None
    credential_ref: str | None


@dataclass
class MigrationPlan:
    roots: list[RootUpdate] = field(default_factory=list)
    owners: dict[str, dict[str, str]] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    unresolved: dict[str, list[str]] = field(default_factory=dict)

    def report(self, *, applied: bool) -> dict[str, Any]:
        return {
            "applied": applied,
            "pending_updates": 0
            if applied
            else len(self.roots) + sum(map(len, self.owners.values())),
            "issues": [asdict(issue) for issue in self.issues],
            "unresolved_sql": self.unresolved,
            "vector_migration_complete": False,
            "ready_for_querying": False,
        }


def _origin(
    row: dict[str, Any],
    override: IdentityOverride,
    reference: str | None,
    issues: list[Issue],
) -> Origin | None:
    previous = row["configuration_origin"]
    if override.origin_specified:
        origin = override.configuration_origin
    elif previous:
        origin = Origin(previous)
    elif row["has_inline"] or reference:
        origin = Origin.MANAGED
    else:
        issues.append(Issue("origin_required", (row["id"],)))
        return None
    if (previous and origin != previous) or (
        origin != Origin.MANAGED and (row["has_inline"] or reference)
    ):
        issues.append(Issue("origin_conflict", (row["id"],)))
    if origin == Origin.MANAGED and not (row["has_inline"] or reference):
        issues.append(Issue("credential_reference_required", (row["id"],)))
    return origin


def _root_update(
    row: dict[str, Any], override: IdentityOverride, issues: list[Issue]
) -> RootUpdate:
    reference = override.credential_ref or row["credential_ref"]
    if row["credential_ref"] and reference != row["credential_ref"]:
        issues.append(Issue("credential_reference_conflict", (row["id"],)))
    label = _nonblank(override.display_label or row["display_label"] or row["name"])
    return RootUpdate(
        row["id"], label, _origin(row, override, reference, issues), reference
    )


def _plan_roots(
    conn: psycopg.Connection, mapping: dict[str, IdentityOverride], plan: MigrationPlan
) -> None:
    # Never select the connection JSON or any secret value for ownership discovery.
    rows = conn.execute(
        "SELECT id, name, display_label, configuration_origin, credential_ref, "
        "connection IS NOT NULL AS has_inline FROM catalog_database ORDER BY id"
    ).fetchall()
    unknown = set(mapping) - {row["id"] for row in rows}
    if unknown:
        plan.issues.append(Issue("unknown_source", tuple(sorted(unknown))))
    labels: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        update = _root_update(
            row, mapping.get(row["id"], IdentityOverride()), plan.issues
        )
        key = conn.execute(
            "SELECT lower(TRIM(BOTH FROM %s::text)) AS key", (update.display_label,)
        ).fetchone()["key"]
        labels[key].append(update.id)
        if any(row[key] != value for key, value in asdict(update).items()):
            plan.roots.append(update)
    for ids in labels.values():
        if len(ids) > 1:
            plan.issues.append(Issue("label_conflict", tuple(ids)))


def _source_owner(
    row: dict[str, Any], dependencies: list[str | None], plan: MigrationPlan, table: str
) -> str | None:
    existing = row["connection_id"]
    if existing:
        if any(owner != existing for owner in dependencies):
            plan.issues.append(Issue("ownership_conflict", (row["id"],)))
        return existing
    owners = set(dependencies)
    if len(owners) == 1 and None not in owners:
        return next(iter(owners))
    plan.unresolved[table].append(row["id"])
    return None


def _plan_queries(
    conn: psycopg.Connection, plan: MigrationPlan
) -> dict[str, str | None]:
    rows = conn.execute(
        "WITH owners AS ("
        "SELECT link.sql_query_id, s.database_id FROM sql_query__table link "
        "JOIN catalog_table t ON t.id = link.table_id "
        "JOIN catalog_schema s ON s.id = t.schema_id UNION "
        "SELECT link.sql_query_id, s.database_id FROM sql_query__column link "
        "JOIN catalog_column c ON c.id = link.column_id "
        "JOIN catalog_table t ON t.id = c.table_id "
        "JOIN catalog_schema s ON s.id = t.schema_id) "
        "SELECT q.id, q.connection_id, "
        "array_remove(array_agg(DISTINCT owners.database_id), NULL) AS owners "
        "FROM sql_query q LEFT JOIN owners ON owners.sql_query_id = q.id "
        "GROUP BY q.id, q.connection_id ORDER BY q.id"
    ).fetchall()
    result: dict[str, str | None] = {}
    plan.owners["sql_query"] = {}
    plan.unresolved["sql_query"] = []
    for row in rows:
        owner = _source_owner(row, row["owners"], plan, "sql_query")
        result[row["id"]] = owner
        if owner is not None and owner != row["connection_id"]:
            plan.owners["sql_query"][row["id"]] = owner
    return result


def _plan_definitions(
    conn: psycopg.Connection, plan: MigrationPlan, query_owners: dict[str, str | None]
) -> None:
    for table, link, key in (
        ("sql_attribute", "sql_attribute__sql", "attribute_id"),
        ("custom_analysis", "custom_analysis__sql", "analysis_id"),
    ):
        rows = conn.execute(
            sql.SQL(
                "SELECT a.id, a.connection_id, "
                "array_remove(array_agg(DISTINCT link.sql_query_id), NULL) AS queries "
                "FROM {} a LEFT JOIN {} link ON link.{} = a.id "
                "GROUP BY a.id, a.connection_id ORDER BY a.id"
            ).format(sql.Identifier(table), sql.Identifier(link), sql.Identifier(key))
        ).fetchall()
        plan.owners[table] = {}
        plan.unresolved[table] = []
        for row in rows:
            dependencies = [query_owners.get(query_id) for query_id in row["queries"]]
            owner = _source_owner(row, dependencies, plan, table)
            if owner is not None and owner != row["connection_id"]:
                plan.owners[table][row["id"]] = owner


def _apply(conn: psycopg.Connection, plan: MigrationPlan) -> None:
    for root in plan.roots:
        conn.execute(
            "UPDATE catalog_database SET display_label = %s, "
            "configuration_origin = %s, credential_ref = %s WHERE id = %s",
            (
                root.display_label,
                root.configuration_origin,
                root.credential_ref,
                root.id,
            ),
        )
    for table, updates in plan.owners.items():
        for node_id, owner in updates.items():
            conn.execute(
                sql.SQL("UPDATE {} SET connection_id = %s WHERE id = %s").format(
                    sql.Identifier(table)
                ),
                (owner, node_id),
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--mapping", type=Path)
    args = parser.parse_args(argv)
    try:
        mapping = validate_mapping(
            json.loads(args.mapping.read_text()) if args.mapping else {}
        )
        with psycopg.connect(
            get_postgres_connection_string(), connect_timeout=3, row_factory=dict_row
        ) as conn:
            conn.read_only = args.check
            if args.apply:
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('auto-ontology:identity-migration'))"
                )
            plan = MigrationPlan()
            _plan_roots(conn, mapping, plan)
            _plan_definitions(conn, plan, _plan_queries(conn, plan))
            vectors = inspect_vectors(conn)
            for table in vectors.collections.values():
                if table.conflicts:
                    plan.issues.append(
                        Issue("vector_identity_conflict", tuple(table.conflicts))
                    )
            if args.apply and not plan.issues:
                _apply(conn, plan)
                apply_vectors(conn, vectors)
        applied = args.apply and not plan.issues
        failures = []
        if applied:
            # Model calls cannot share the atomic metadata transaction. A failed
            # rebuild leaves legacy semantic rows unscoped and safely retryable.
            with psycopg.connect(
                get_postgres_connection_string(), row_factory=dict_row
            ) as conn:
                conn.read_only = True
                vectors = inspect_vectors(conn)
            for owner in vectors.missing_semantic():
                try:
                    if not rebuild_semantic(owner, vectors.sources[owner]):
                        failures.append(owner)
                except Exception:
                    failures.append(owner)
            with psycopg.connect(
                get_postgres_connection_string(), row_factory=dict_row
            ) as conn:
                conn.read_only = True
                vectors = inspect_vectors(conn)
            failures = sorted(set(failures) | set(vectors.missing_semantic()))
        report = {
            **plan.report(applied=applied),
            **vectors.report(),
            "semantic_rebuild_failed": failures,
        }
        if plan.issues or failures:
            report["ready_for_querying"] = False
        if failures:
            report["vector_migration_complete"] = False
        print(json.dumps(report))
        return 2 if plan.issues or failures else 0
    except (ValueError, OSError, KeyError, psycopg.Error) as exc:
        # Driver/config errors can echo URLs or secret-bearing values.
        print(
            f"Identity migration failed ({type(exc).__name__}); check schema and mapping.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
