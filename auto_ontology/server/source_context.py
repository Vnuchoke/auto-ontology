# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional source context for curated reads; omission preserves the global view."""

from fastapi import HTTPException

from auto_ontology.connectors import sources


def resolve_semantic_source(
    connection_id: str | None = None,
    connection_label: str | None = None,
    target_db: str | None = None,
) -> str | None:
    if all(value is None for value in (connection_id, connection_label, target_db)):
        return None
    try:
        return sources.resolve_source(
            connection_id=connection_id,
            connection_label=connection_label,
            target_db=target_db,
        ).connection_id
    except sources.SourceSelectionError as exc:
        raise HTTPException(exc.status_code, detail={"code": exc.code}) from None
