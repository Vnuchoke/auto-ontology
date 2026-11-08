# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row upserts — the one write primitive the catalog pipeline uses.

Every writer needs the same thing: *find a row by its identity columns, create
it with these values if absent, update it if present*.

Roots are keyed by their registered ID. Descendants converge on their natural
key within that root: ``Schema``, ``Table`` and ``Column`` are unique *within
their parent*, which is why :func:`upsert_row` takes ``parent_id``. Existing IDs
must agree with that parent; a caller cannot reassign a foreign-source node.

**One deliberate divergence: the stored id is never overwritten.** In a property
graph, ``id`` is an ordinary property and relationships bind to internal nodes,
so replacing it costs nothing. Here it is the primary key with other rows
referencing it, and overwriting it violates their foreign keys — a real
``IntegrityError``, not a theoretical one.

Nothing is lost by not reproducing it. The parser resolves ids out of the store
before writing (``get_table_ids`` / ``get_column_ids``), so for ``Table`` and
``Column`` the incoming id already *is* the stored one. Only a node matched on
something else — a ``Sql`` matched by its statement text — arrives with a freshly
generated id, and there keeping the stored id is the correct answer rather than
merely the safe one.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Text, and_, literal, select
from sqlalchemy.dialects.postgresql import insert

from auto_ontology.catalog.store import registry
from auto_ontology.dal.connections import (
    assert_connection_id_available,
    lock_source_registration,
)
from auto_ontology.dal.session import store, write_transaction

logger = logging.getLogger(__name__)


def _natural_key_predicate(
    spec: registry.EntitySpec, values: dict[str, Any], parent_id: str | None
):
    """Build the WHERE that decides whether this row already exists."""
    clauses = []
    for part in spec.natural_key:
        if part == "parent":
            if spec.parent_column is None:
                raise ValueError(f"{spec.label} has no parent column to match on")
            clauses.append(spec.table.c[spec.parent_column] == parent_id)
            continue

        column = spec.table.c[part]
        value = values.get(part)
        transform = spec.key_expressions.get(part)
        if transform is not None and value is not None:
            # Both sides through the same function, so the expression index is
            # usable -- Postgres matches an expression index syntactically, and
            # `col = $1` does not match an index on `md5(col)`. The `md5` on the
            # bind side is a per-row cost, not a per-table one.
            #
            # `value is not None` because `md5(NULL) = md5(NULL)` is NULL, not
            # true; the direct comparison below renders `IS NULL`, which is what
            # a caller matching on a missing value means.
            clauses.append(transform(column) == transform(literal(value, Text)))
        else:
            clauses.append(column == value)
    return and_(*clauses)


def resolve_id(
    label: str,
    match_props: dict[str, Any],
    *,
    parent_id: str | None = None,
) -> str | None:
    """Find an existing row's id from whatever the caller matched on.

    Roots require an ID, schemas a connection_id (or parent_id), and descendants
    their parent ID. An existing ID never bypasses a supplied ancestry check;
    physical names are not source selectors.
    """
    spec = registry.entity_spec(label)

    if "connection_id" in spec.natural_key and not match_props.get("connection_id"):
        raise ValueError("SQL identity requires connection_id")

    if spec.label == registry.Labels.DB and not match_props.get("id"):
        raise ValueError("Database identity requires connection_id as id")
    if spec.parent_label == registry.Labels.DB:
        source_id = match_props.get("connection_id")
        if source_id and parent_id and source_id != parent_id:
            raise ValueError("Schema connection_id conflicts with parent")
        parent_id = source_id or parent_id
        if parent_id is None and not match_props.get("id"):
            raise ValueError("Schema identity requires a connection_id or parent")

    if "id" in match_props and match_props["id"]:
        identity_columns = [spec.table.c.id]
        if "connection_id" in spec.natural_key:
            identity_columns.append(spec.table.c.connection_id)
        if spec.parent_column:
            identity_columns.append(spec.table.c[spec.parent_column])
        if "name" in spec.columns:
            identity_columns.append(spec.table.c.name)
        rows = store().query_read(
            select(*identity_columns).where(spec.table.c.id == match_props["id"])
        )
        if rows:
            if (
                "connection_id" in spec.natural_key
                and rows[0]["connection_id"] != match_props["connection_id"]
            ):
                raise ValueError(
                    "SQL ID belongs to another source or unresolved legacy"
                )
            if (
                spec.parent_column
                and parent_id is not None
                and rows[0][spec.parent_column] != parent_id
            ):
                raise ValueError("Catalog ID belongs to another parent/source")
            if (
                match_props.get("name") is not None
                and rows[0].get("name") != match_props["name"]
            ):
                raise ValueError("Catalog ID conflicts with the supplied name")
            return rows[0]["id"]
        # Fall through: a parser-generated id that is not in the store yet is
        # the normal case for a new node, so identity falls back to the
        # natural key rather than declaring the row absent.

    if spec.parent_column and parent_id is None:
        return None

    if not spec.natural_key:
        return None

    rows = store().query_read(
        select(spec.table.c.id).where(
            _natural_key_predicate(spec, match_props, parent_id)
        )
    )
    return rows[0]["id"] if rows else None


def upsert_row(
    label: str | list[str] | tuple[str, ...],
    match_props: dict[str, Any],
    properties: dict[str, Any],
    *,
    parent_id: str | None = None,
    on_match: dict[str, Any] | None = None,
) -> str:
    """Upsert a row; root identity checks and writes share the registration lock."""
    label = registry.entity_spec(label).label
    if label == registry.Labels.DB:
        with write_transaction():
            lock_source_registration()
            if match_props.get("id"):
                assert_connection_id_available(match_props["id"])
            return _upsert_row(
                label, match_props, properties, parent_id=parent_id, on_match=on_match
            )
    return _upsert_row(
        label, match_props, properties, parent_id=parent_id, on_match=on_match
    )


def _upsert_row(
    label: str,
    match_props: dict[str, Any],
    properties: dict[str, Any],
    *,
    parent_id: str | None = None,
    on_match: dict[str, Any] | None = None,
) -> str:
    """Create or update one row, returning its id.

    Find-or-create, then update:
    properties are written on insert, and *on_match* is applied when the row
    already existed — except for ``id``, which is never overwritten. See the
    module docstring for why that divergence is both necessary and free.
    """
    spec = registry.entity_spec(label)
    values = registry.projected(label, properties)
    if "connection_id" in spec.natural_key:
        owner = match_props.get("connection_id")
        if not owner or values.get("connection_id") != owner:
            raise ValueError("SQL properties require the matched connection_id")
        if (on_match or {}).get("connection_id", owner) != owner:
            raise ValueError("SQL ownership cannot change")
    if spec.label == registry.Labels.DB and values.get("id") != match_props.get("id"):
        raise ValueError("Database properties must carry the matched connection_id")

    if spec.parent_label == registry.Labels.DB:
        source_id = match_props.get("connection_id")
        if source_id and parent_id and source_id != parent_id:
            raise ValueError("Schema connection_id conflicts with parent")
        parent_id = source_id or parent_id
    if spec.parent_column and parent_id is None:
        parent_id = values.get(spec.parent_column)
    if spec.parent_column and parent_id is not None:
        if values.get(spec.parent_column) not in (None, parent_id):
            raise ValueError("Catalog properties conflict with supplied parent/source")
    incoming_id = values.get("id")
    if incoming_id and incoming_id != match_props.get("id"):
        # Properties must not smuggle an existing foreign ID past match_props.
        resolve_id(
            spec.label,
            {
                "id": incoming_id,
                **(
                    {"connection_id": values["connection_id"]}
                    if "connection_id" in spec.natural_key
                    else {}
                ),
            },
            parent_id=parent_id,
        )

    if spec.parent_column and parent_id is not None:
        values[spec.parent_column] = parent_id

    existing_id = resolve_id(label, match_props, parent_id=parent_id)
    if spec.parent_column and (on_match or {}).get(spec.parent_column) not in (
        None,
        parent_id,
    ):
        raise ValueError("Catalog update conflicts with supplied parent/source")

    if existing_id is not None and spec.label == registry.Labels.DB:
        roots = store().query_read(
            select(spec.table.c.name).where(spec.table.c.id == existing_id)
        )
        if values.get("name") not in (None, roots[0]["name"]):
            raise ValueError("Database name conflicts with registered connection_id")
        # Connection metadata belongs exclusively to bootstrap/lifecycle, not ingest.
        on_match = {k: v for k, v in (on_match or {}).items() if k == "pulled"}

    if existing_id is None:
        # Let the natural key arbitrate a race rather than checking first and
        # inserting second: two ingestion threads can reach here at once.
        conflict_target = _conflict_target(spec)
        statement = insert(spec.table).values(**values)
        if conflict_target:
            skip = (
                set(spec.natural_key)
                | {spec.parent_column if p == "parent" else p for p in spec.natural_key}
                # `id` is the primary key. Updating it on conflict rewrites the
                # existing row's identity, dangling any id the caller already
                # holds -- the same reason the match branch below refuses to
                # overwrite it.
                | {"id"}
            )
            update = (
                {}
                if spec.label == registry.Labels.DB
                else {k: v for k, v in values.items() if k not in skip}
            )
            statement = (
                statement.on_conflict_do_update(
                    index_elements=conflict_target, set_=update
                )
                if update
                else statement.on_conflict_do_nothing(index_elements=conflict_target)
            )
        rows = store().query_write(statement.returning(spec.table.c.id))
        if rows:
            return rows[0]["id"]
        # No RETURNING row means DO NOTHING fired: someone else won the race
        # and their row is the one that exists. Read it back.
        raced_id = resolve_id(label, match_props, parent_id=parent_id)
        if raced_id is None:
            # Not reachable through a race -- the conflict proves a row is
            # there. It means the natural key cannot find what the conflict
            # target matched, i.e. the two disagree in the registry. Returning
            # None here would satisfy the `-> str` annotation in name only and
            # surface as a NOT NULL violation on the *next* insert, pointing at
            # an innocent child row.
            raise RuntimeError(
                f"{label}: insert conflicted but the row is not resolvable by "
                f"its natural key {sorted(spec.natural_key)} -- the ON CONFLICT "
                "target and the natural key disagree"
            )
        return raced_id

    incoming_id = properties.get("id")
    if incoming_id and incoming_id != existing_id:
        # An incoming id is never written over an existing row's. That is safe in a
        # property graph, where relationships bind to internal nodes and `id` is
        # just a property — but here `id` is the primary key and other rows
        # reference it, so overwriting it violates their foreign keys.
        #
        # Not reproducing it loses nothing: the parser resolves ids out of the
        # store before writing (get_table_ids / get_column_ids), so for Table
        # and Column the id it carries is already the one on the row. Only
        # nodes matched on something else — a Sql matched by its statement text
        # — arrive with a freshly generated id, and for those keeping the
        # stored id is the correct answer, not merely the safe one.
        logger.debug(
            "keeping stored id %s for %s rather than the incoming %s",
            existing_id,
            label,
            incoming_id,
        )

    update = registry.projected(label, dict(on_match or {}))
    update.pop("id", None)
    if spec.parent_column and parent_id is not None:
        update[spec.parent_column] = parent_id

    if update:
        store().query_write(
            spec.table.update().where(spec.table.c.id == existing_id).values(**update)
        )
        return existing_id
    return existing_id


def _conflict_target(spec: registry.EntitySpec) -> list:
    """What ``ON CONFLICT`` should name for this label.

    Usually the natural key's columns, but ``sql_query`` is unique on
    ``md5(sql_full_query)`` rather than the column itself — naming the column
    there raises "no unique or exclusion constraint matching the ON CONFLICT
    specification", because Postgres matches the target against an index.
    """
    if spec.conflict_expression is not None:
        return [spec.table.c.connection_id, spec.conflict_expression]
    if not spec.natural_key:
        return []
    columns = [
        spec.parent_column if part == "parent" else part for part in spec.natural_key
    ]
    return [c for c in columns if c]
