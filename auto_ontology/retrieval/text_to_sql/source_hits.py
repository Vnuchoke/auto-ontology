# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify vector identifiers against persisted ownership before model consumption."""

from auto_ontology.catalog.constants import Labels
from auto_ontology.dal.attributes import fetch_attr_column_contexts
from auto_ontology.dal.custom_analyses import fetch_custom_analyses_with_sql
from auto_ontology.dal.sql_attributes import fetch_sql_attributes_with_sql
from auto_ontology.dal.terms import get_full_term_by_id
from auto_ontology.semantic.constants import (
    LABEL_COLUMN_ATTRIBUTE,
    LABEL_SQL_ATTRIBUTE,
    LABEL_TERM,
)


def source_owned_hits(
    hits: list[dict], label: str, *, connection_id: str
) -> list[dict]:
    """Drop forged/stale vector identities; metadata alone is not ownership proof."""
    hits = [
        hit
        for hit in hits
        if hit.get("connection_id") == connection_id and hit.get("id")
    ]
    ids = list(dict.fromkeys(str(hit["id"]) for hit in hits))
    if not ids:
        return []
    if label == LABEL_COLUMN_ATTRIBUTE:
        allowed = set(fetch_attr_column_contexts(ids, connection_id=connection_id))
    elif label == LABEL_SQL_ATTRIBUTE:
        allowed = {
            str(row["id"])
            for row in fetch_sql_attributes_with_sql(ids, connection_id=connection_id)
        }
    elif label == Labels.CUSTOM_ANALYSIS:
        allowed = {
            str(row["id"])
            for row in fetch_custom_analyses_with_sql(ids, connection_id=connection_id)
        }
    elif label == LABEL_TERM:
        allowed = {
            node_id
            for node_id in ids
            if get_full_term_by_id(node_id, connection_id=connection_id)
        }
    else:
        raise ValueError("Unsupported source-bound semantic label.")
    return [hit for hit in hits if str(hit["id"]) in allowed]
