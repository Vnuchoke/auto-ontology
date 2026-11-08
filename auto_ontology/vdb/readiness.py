# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only minimum query gate; never constructs a VDB or creates collections."""

import psycopg
from psycopg import sql

from auto_ontology.infra.postgres import get_postgres_connection_string
from auto_ontology.vdb import DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION, VDB_SCHEMA


def require_source_vectors_ready(connection_id: str) -> None:
    """Require indexed CID columns and selected-source rows in both collections.

    This is a runtime prerequisite, not certification of complete vector coverage.
    Global PQL and quarantined NULL-CID rows cannot make a source ready.
    """
    if not isinstance(connection_id, str) or not connection_id.strip():
        raise ValueError("source_not_ready")
    try:
        with psycopg.connect(
            get_postgres_connection_string(),
            options="-c default_transaction_read_only=on",
        ) as conn:
            for collection in (DATA_VDB_COLLECTION, SEMANTIC_VDB_COLLECTION):
                indexed = conn.execute(
                    """SELECT EXISTS (
                        SELECT 1 FROM pg_catalog.pg_class t
                        JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
                        JOIN pg_catalog.pg_attribute a ON a.attrelid = t.oid
                        JOIN pg_catalog.pg_index i ON i.indrelid = t.oid
                        WHERE n.nspname = %s AND t.relname = %s
                          AND a.attname = 'connection_id' AND NOT a.attisdropped
                          AND a.atttypid = 'text'::regtype
                          AND i.indisvalid AND i.indisready
                          AND i.indpred IS NULL
                          AND a.attnum = ANY((i.indkey::smallint[])[0:i.indnkeyatts - 1])
                    )""",
                    (VDB_SCHEMA, collection),
                ).fetchone()
                if not indexed or not indexed[0]:
                    raise ValueError("source_not_ready")
                present = conn.execute(
                    sql.SQL("SELECT 1 FROM {} WHERE connection_id = %s LIMIT 1").format(
                        sql.Identifier(VDB_SCHEMA, collection)
                    ),
                    (connection_id,),
                ).fetchone()
                if present is None:
                    raise ValueError("source_not_ready")
    except psycopg.Error:
        raise ValueError("source_not_ready") from None
