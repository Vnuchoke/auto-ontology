# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads over the ``conversation`` table.

The vector store holds a copy of each turn's text, but it is the *embedding*
text -- a header plus the pair, shaped for a similarity match. The rows here
are the transcript as it was said, and that is what should reach a model.
Reading them back also means retrieval can return a turn nobody searched for:
a neighbour pulled in to give a hit its context has no vector of its own to
come from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, and_, func, or_, select

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store

__all__ = ["fetch_turns", "turn_count"]

# Columns every read returns, in transcript order. Named once so a caller can
# rely on the shape without the query being repeated in three places.
_COLUMNS = (
    s.conversation.c.id,
    s.conversation.c.conversation_id,
    s.conversation.c.turn_index,
    s.conversation.c.session_index,
    s.conversation.c.question,
    s.conversation.c.answer,
    s.conversation.c.time_anchor,
)


def memory_connection_id(database_name: str) -> str | None:
    """Resolve a legacy standalone memory name, never a configured source."""
    root = s.catalog_database
    rows = store().query_read(
        select(root.c.id).where(
            root.c.name == database_name,
            root.c.configuration_origin.is_(None),
        )
    )
    if len(rows) > 1:
        raise ValueError("ambiguous_source: select the conversation connection_id")
    return str(rows[0]["id"]) if rows else None


def _scoped(database_name: str, connection_id: str | None) -> Select[Any]:
    """The join that scopes a read to one memory."""
    return select(*_COLUMNS).where(
        s.conversation.c.database_id
        == (connection_id or memory_connection_id(database_name))
    )


def fetch_turns(
    database_name: str,
    *,
    spans: Sequence[tuple[str, int, int]] | None = None,
    connection_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read turns from one memory, ordered as the conversation ran.

    *spans* is a sequence of ``(conversation_id, low, high)`` windows, each
    inclusive; omit it to read everything. Windows rather than a list of
    indexes because the caller that needs this is widening a hit to its
    neighbours, and a range is both what that means and one predicate instead
    of one per turn.

    Several windows per conversation are expected and are not merged here --
    two hits at opposite ends of a transcript are two windows, and collapsing
    them into one would read every turn in between.

    Always ordered by ``(conversation_id, turn_index)``. Transcript order is
    the only order these rows are useful in -- a reader has to know what was
    said before what -- so it is not left to the caller to remember.
    """
    query = _scoped(database_name, connection_id)
    if spans is not None:
        if not spans:
            return []
        query = query.where(
            or_(
                *[
                    and_(
                        s.conversation.c.conversation_id == conversation_id,
                        s.conversation.c.turn_index >= low,
                        s.conversation.c.turn_index <= high,
                    )
                    for conversation_id, low, high in spans
                ]
            )
        )
    query = query.order_by(
        s.conversation.c.conversation_id, s.conversation.c.turn_index
    )
    return store().query_read(query)


def turn_count(database_name: str, *, connection_id: str | None = None) -> int:
    """How many turns this memory holds."""
    rows = store().query_read(
        select(func.count().label("turns")).select_from(
            _scoped(database_name, connection_id).subquery()
        )
    )
    return int(rows[0]["turns"]) if rows else 0
