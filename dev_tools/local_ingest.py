# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local ingest: docker-compose Postgres source → pgvector embeddings store.

Run after ``docker compose up -d`` and ``dev_tools.fixtures.seed_local_postgres``.

Usage::

    uv run --no-sync python -m dev_tools.local_ingest
"""

from __future__ import annotations

import logging
from auto_ontology.connectors.registry import get_connector
from auto_ontology.connectors.sources import bootstrap_sources, list_active_sources
from auto_ontology.dal.schema_version import require_current_schema
from auto_ontology.ingestion_service.ingest import run_ingest as ingest_connector

logger = logging.getLogger("dev_tools.local_ingest")


def run_ingest(connection_id: str) -> None:
    """Ingest only a registered, active source through the shared pipeline."""
    ingest_connector(get_connector(connection_id))


def main() -> None:
    """Register env metadata, then ingest the first configured source as before."""
    require_current_schema()
    bootstrap_sources()
    sources = list_active_sources()
    if not sources:
        raise EnvironmentError("No active sources configured for local ingest")
    run_ingest(sources[0].connection_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        main()
    except KeyboardInterrupt:
        logger.info("local_ingest: shutting down")
        raise SystemExit(0)
