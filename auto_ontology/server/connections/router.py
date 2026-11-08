# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API routes for UI-managed database connections."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Self

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, model_validator

from auto_ontology.connectors import sources
from auto_ontology.dal.connections import list_connections as _list_connections
from auto_ontology.server.connections import service
from auto_ontology.server.models import SourceSummary
from auto_ontology.server.responses import (
    ListResponse,
    ConnectionListResponse,
    ConnectionResponse,
    ConnectionTestResponse,
    SsoFederationResponse,
)

router = APIRouter()


class ConnectionBody(BaseModel):
    connection: dict[str, Any]
    display_label: str | None = None


class ConnectionUpdateBody(BaseModel):
    connection: dict[str, Any] | None = None
    display_label: str | None = None

    @model_validator(mode="after")
    def require_update(self) -> Self:
        if self.connection is None and self.display_label is None:
            raise ValueError("Provide connection settings or display_label")
        return self


class ConnectionTestBody(ConnectionBody):
    replacing: str | None = None
    replacing_connection_id: str | None = None

    @model_validator(mode="after")
    def require_one_selector(self) -> Self:
        if self.replacing is not None and self.replacing_connection_id is not None:
            raise ValueError("invalid_source_selector")
        return self


@contextmanager
def _connection_errors() -> Iterator[None]:
    """Keep lifecycle status codes consistent without exposing backend exceptions."""
    try:
        yield
    except service.LabelConflictError:
        raise HTTPException(409, "label_conflict") from None
    except service.AmbiguousConnectionError:
        raise HTTPException(409, "ambiguous_source: use connection_id") from None
    except service.SourceUnavailableError:
        raise HTTPException(503, "source_unavailable") from None
    except LookupError:
        raise HTTPException(404, "source_not_found") from None
    except ValueError as exc:
        if str(exc).startswith("source_unavailable"):
            raise HTTPException(503, "source_unavailable") from None
        raise HTTPException(422, str(exc)) from None
    except Exception:
        raise HTTPException(422, "Connection operation failed") from None


def _serialize_connection(row: dict[str, Any]) -> dict[str, Any]:
    """Expose identity separately from credential-free driver settings."""
    result = {
        key: row[key] for key in ("connection_id", "display_label", "database_name")
    }
    if "connection" in row:
        result["connection"] = {
            key: value
            for key, value in row["connection"].items()
            if key not in service.SECRET_FIELDS | {"credential_ref"}
        }
    return result


@router.get("/connections/source", response_model=bool)
def is_env_source() -> bool:
    """Report whether connections are managed via the ``CONNECTION_STRINGS`` env."""
    connection_strings = os.environ.get("CONNECTION_STRINGS", "")
    connection_strings_exists = [
        cs.strip() for cs in connection_strings.split(",") if cs.strip()
    ]
    return bool(connection_strings_exists)


@router.get("/connections", response_model=ConnectionListResponse)
def list_connections() -> dict:
    """List every configured database connection, credential-free.

    ID and label are separate from the physical database name. Credentials and
    credential references are never returned; this settings route retains its
    existing permissions (chat discovery is a separate endpoint).
    """
    with _connection_errors():
        rows = [_serialize_connection(conn) for conn in _list_connections()]
    return {"data": rows, "count": len(rows)}


@router.get("/connections/sources", response_model=ListResponse[SourceSummary])
def list_sources() -> dict:
    """List active source identities without credentials or source-driver probes."""
    try:
        active = sources.list_active_sources()
    except Exception:
        raise HTTPException(503, "source_not_ready") from None
    rows = [
        SourceSummary(
            connection_id=item.connection_id,
            display_label=item.display_label,
            database_name=item.database_name,
            dialect=item.dialect,
        )
        for item in active
    ]
    return {"data": rows, "count": len(rows)}


@router.post("/connections/test", response_model=ConnectionTestResponse)
def test_connection(body: ConnectionTestBody) -> dict:
    """Validate a connection and return its schemas (for the schema picker).

    ``replacing_connection_id`` retains blank credentials from exactly that
    source. Legacy ``replacing`` accepts only an unambiguous physical name.
    Supplying both selectors is invalid. New sources may share physical names.
    """
    with _connection_errors():
        schemas = service.test_connection(
            body.connection,
            replacing=body.replacing,
            replacing_connection_id=body.replacing_connection_id,
        )
    return {"success": True, "schemas": schemas}


@router.post("/connections", status_code=201, response_model=ConnectionResponse)
def create_connection(body: ConnectionBody) -> dict:
    """Register a database connection and return it credential-free.

    An omitted label defaults to the physical name only if free. Blank labels
    are invalid; normalized label conflicts return 409 without replacing an
    existing source. Test credentials separately with ``POST /connections/test``.
    """
    with _connection_errors():
        row = service.create_connection(
            connection=body.connection, display_label=body.display_label
        )

    return {"data": _serialize_connection(row)}


@router.put("/connections/{database_name}", response_model=ConnectionResponse)
def update_connection(
    body: ConnectionUpdateBody,
    database_name: str = Path(
        description=("Legacy physical name; must identify exactly one managed source.")
    ),
) -> dict:
    """Legacy update: ambiguous names return 409 before any mutation.

    Prefer the by-id endpoint. Blank secrets retain this source's credentials;
    label-only updates do not test connectivity or trigger ingestion.
    """
    with _connection_errors():
        row = service.update_connection(
            database_name=database_name,
            connection=body.connection,
            display_label=body.display_label,
        )

    return {"data": _serialize_connection(row)}


class SsoFederationBody(BaseModel):
    enabled: bool


def _serialize_sso(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("connection_id", "display_label", "database_name", "sso_federation")
    }


@router.patch(
    "/connections/{database_name}/sso-federation",
    response_model=SsoFederationResponse,
)
def set_sso_federation(
    body: SsoFederationBody,
    database_name: str = Path(
        description=("Legacy physical name; must identify exactly one managed source.")
    ),
) -> dict:
    """Legacy SSO toggle; ambiguous names return 409 without credential writes."""
    with _connection_errors():
        row = service.set_sso_federation(
            database_name=database_name, enabled=body.enabled
        )

    return {"data": _serialize_sso(row)}


@router.delete(
    "/connections/{database_name}",
    response_model=ConnectionResponse,
    response_model_exclude_unset=True,
)
def delete_connection(
    database_name: str = Path(
        description=("Legacy physical name; must identify exactly one managed source.")
    ),
) -> dict:
    """Legacy removal: ambiguous names return 409 before any deletion.

    The response echoes the captured ID, not just a reusable name. Cleanup of
    ingested data is asynchronous; success retires the connection, not its history.
    """
    with _connection_errors():
        row = service.delete_connection(database_name)

    if row is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    return {"data": _serialize_connection(row)}


@router.put("/connections/by-id/{connection_id}", response_model=ConnectionResponse)
def update_connection_by_id(connection_id: str, body: ConnectionUpdateBody) -> dict:
    """Update this managed ID; physical database and driver type cannot change.

    Send only display_label for a metadata-only rename. Blank secrets retain
    this ID's stored values. Label conflicts return 409, unknown IDs 404, and
    unavailable credentials 503; another source is never substituted.
    """
    with _connection_errors():
        row = service.update_connection(
            connection_id=connection_id,
            connection=body.connection,
            display_label=body.display_label,
        )
    return {"data": _serialize_connection(row)}


@router.post(
    "/connections/by-id/{connection_id}/test", response_model=ConnectionTestResponse
)
def test_connection_by_id(connection_id: str, body: ConnectionBody) -> dict:
    """Test this source's proposed settings, retaining its own blank secrets."""
    with _connection_errors():
        schemas = service.test_connection(
            body.connection, replacing_connection_id=connection_id
        )
    return {"success": True, "schemas": schemas}


@router.patch(
    "/connections/by-id/{connection_id}/sso-federation",
    response_model=SsoFederationResponse,
)
def set_sso_federation_by_id(connection_id: str, body: SsoFederationBody) -> dict:
    """Toggle selected-source SSO without changing credentials of other IDs."""
    with _connection_errors():
        row = service.set_sso_federation(
            connection_id=connection_id, enabled=body.enabled
        )
    return {"data": _serialize_sso(row)}


@router.delete(
    "/connections/by-id/{connection_id}",
    response_model=ConnectionResponse,
    response_model_exclude_unset=True,
)
def delete_connection_by_id(connection_id: str) -> dict:
    """Retire only this ID and enqueue its captured ID for data cleanup.

    History remains bound to the retired ID. Missing IDs return 404; failed
    credential removal returns 503, without deleting a same-name neighbor.
    """
    with _connection_errors():
        row = service.delete_connection(connection_id=connection_id)
    if row is None:
        raise HTTPException(404, "source_not_found")
    return {"data": _serialize_connection(row)}
