# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the ingestion service API.

Lightweight on purpose: it only depends on ``httpx`` so the server can trigger
ingest over HTTP without importing ``auto_ontology.ingestion_service.ingest``
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SERVICE_URL = "http://localhost:3002"

# The service endpoints return 202 immediately, so a short timeout is enough.
_REQUEST_TIMEOUT = 10.0


def _base_url() -> str:
    """Base URL of the ingestion service, configurable for local vs Docker."""
    url = os.environ.get("INGESTION_SERVICE_URL", DEFAULT_INGESTION_SERVICE_URL)
    return url.rstrip("/")


def trigger_ingest(connection_id: str) -> None:
    """Ask the ingestion service to ingest this source ID (best-effort)."""
    url = f"{_base_url()}/ingest"
    try:
        httpx.post(
            url, json={"connection_id": connection_id}, timeout=_REQUEST_TIMEOUT
        ).raise_for_status()
    except Exception:
        logger.exception("Failed to trigger ingest via %s", url)


def trigger_semantic_compile() -> None:
    """Ask the ingestion service to run semantic compilation (best-effort).

    The service starts its semantic scheduler if it isn't running yet, so this
    also takes effect when compilation is enabled while the service is up.
    """
    url = f"{_base_url()}/semantic/compile"
    try:
        httpx.post(url, timeout=_REQUEST_TIMEOUT).raise_for_status()
    except Exception:
        logger.exception("Failed to trigger semantic compile via %s", url)


def is_semantic_compilation_running() -> bool:
    """Ask the ingestion service whether a compilation pass is executing right now.

    This is the *only* part of semantic compilation status that genuinely
    requires the live ingestion process — "running" is in-memory state on its
    scheduler. ``last_success_at``/``last_failure_at`` are plain Postgres reads
    (see ``auto_ontology/ingestion_service/history.py``) that the API server does
    directly instead of relaying through here, precisely so a temporarily
    unreachable ingestion service doesn't also wipe out history that has
    nothing to do with it.

    Best-effort: if the ingestion service can't be reached, this reports
    ``False`` rather than raising, so a transient outage makes the settings
    page look idle instead of stuck. Failures are logged as a one-line warning
    rather than ``logger.exception`` — this is called on every settings-page
    load, so a full traceback each time would flood the logs whenever the
    ingestion service is simply not running (e.g. local dev).
    """
    url = f"{_base_url()}/semantic/status"
    try:
        response = httpx.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return bool(response.json().get("running", False))
    except Exception as exc:
        logger.warning(
            "Failed to query semantic compilation status via %s: %s", url, exc
        )
        return False


def trigger_semantic_reset(connection_id: str | None = None) -> None:
    """Ask the ingestion service to delete the semantic layer (best-effort).

    Omitting ``connection_id`` resets every source; global PQL is preserved.
    The service stops and drains compilation, deletes, then rebuilds. A 202
    acknowledges queueing, not successful completion.
    """
    url = f"{_base_url()}/semantic/reset"
    params = {} if connection_id is None else {"connection_id": connection_id}
    try:
        httpx.post(
            url,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        ).raise_for_status()
    except Exception:
        logger.error(
            "Failed to trigger semantic reset for connection %s", connection_id
        )


def trigger_reset(connection_id: str) -> None:
    """Queue final cleanup of this retired source ID (best-effort, not a reset).

    The caller captures the ID before retiring credentials. Never resolve a
    reusable label or physical name on the asynchronous cleanup path.
    """
    url = f"{_base_url()}/ingest/delete"
    try:
        httpx.post(
            url,
            params={"connection_id": connection_id},
            timeout=_REQUEST_TIMEOUT,
        ).raise_for_status()
    except Exception:
        logger.error("Failed to trigger final cleanup for connection %s", connection_id)
