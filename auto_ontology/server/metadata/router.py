# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API routes for grading entity coverage of a free-text question."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, StrictStr
from auto_ontology.connectors.sources import SourceSelectionError
from auto_ontology.connectors.databricks_oauth import DatabricksOAuthError

from auto_ontology.retrieval.entity_coverage.state import DEFAULT_MAX_DISTANCE
from auto_ontology.server.metadata import service as dal
from auto_ontology.server.responses import EntityCoverageResponse

router = APIRouter()


class EntityCoverageRequest(BaseModel):
    """Payload for the entity-coverage route."""

    question: str = Field(..., min_length=1)
    connection_id: StrictStr | None = None
    connection_label: StrictStr | None = None
    target_db: StrictStr | None = None
    max_distance: float = Field(
        default=DEFAULT_MAX_DISTANCE,
        gt=0.0,
        description=(
            "Maximum L2 vector distance for a candidate to count "
            "(lower score = closer match)."
        ),
    )


@router.post("/question-entity-coverage", response_model=EntityCoverageResponse)
def entity_coverage(body: EntityCoverageRequest, request: Request) -> dict:
    """Return ranked semantic candidates and a 0–1 entity coverage grade.

    Extracts entities from the question, retrieves semantic candidates, filters
    by vector distance, enriches from the catalog, and grades how many entities have
    at least one covering ColumnAttribute candidate.

    Returns 422 when the flow cannot produce a result for the question.
    """
    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    try:
        result = dal.entity_coverage(
            body.question,
            max_distance=body.max_distance,
            connection_id=body.connection_id,
            connection_label=body.connection_label,
            target_db=body.target_db,
            subject_token=token.strip() if scheme.lower() == "bearer" else None,
        )
    except SourceSelectionError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail={"code": exc.code}
        ) from exc
    except DatabricksOAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except dal.PredictionFlowError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"data": result}
