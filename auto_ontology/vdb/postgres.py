# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Postgres + pgvector vector store for the data and semantic layers.

Backed by :class:`langchain_postgres.PGVectorStore` (the v2 vector store API).
Records are carried as :class:`langchain_core.documents.Document` objects
throughout. ``connection_id`` is the promoted source identity; physical
``database_name`` is descriptive metadata, never a routing or deletion key.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Iterable, Optional

import psycopg
from psycopg import sql
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from auto_ontology.vdb.records import (
    global_pql_projection_id,
    projection_id,
    validate_global_pql_metadata,
)

logger = logging.getLogger(__name__)

_DATABASE_METADATA_COLUMN = "database_name"
_LABEL_METADATA_COLUMN = "label"
_CONNECTION_METADATA_COLUMN = "connection_id"


class _UnusableEmbeddings(Embeddings):
    """Placeholder used when no query-side embedder is supplied.

    Ingestion via :meth:`PGVectorStore.add_embeddings` does not call this — it
    only fires if someone tries to run :meth:`PostgresVDB.retrieval` without
    passing an ``embeddings`` instance to the constructor.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "PostgresVDB has no embeddings function configured. "
            "Pass `embeddings=<Embeddings>` to the constructor to enable retrieval."
        )

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _flatten(records: Iterable) -> Iterable[dict]:
    """Yield record dicts from a possibly-nested list of records."""
    for item in records:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _flatten(item)


def _to_async_url(url: str) -> str:
    """Convert a libpq-style URL to an async-SQLAlchemy URL for ``PGEngine``."""
    if "+asyncpg" in url or "+psycopg" in url:
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _instrument_engine(engine: PGEngine) -> None:
    """Time every SQL statement this engine executes.

    Attached to ``sync_engine`` because that is where SQLAlchemy emits events
    even for an async engine.
    """
    try:
        from sqlalchemy import event

        target = getattr(engine, "_pool", None) or getattr(engine, "_engine", None)
        sync_engine = getattr(target, "sync_engine", None) or target
        if sync_engine is None:
            return

        # The connection, cursor, bind parameters and execution context are
        # typed ``Any``: they are DBAPI and SQLAlchemy internals whose concrete
        # types vary by dialect, and the timestamp below is stashed on the
        # context as a private attribute, which a precise type would reject.
        @event.listens_for(sync_engine, "before_cursor_execute")
        def _before(
            conn: Any,
            cursor: Any,
            statement: str,
            parameters: Any,
            context: Any,
            executemany: bool,
        ) -> None:
            context._auto_ontology_started = time.perf_counter()

        @event.listens_for(sync_engine, "after_cursor_execute")
        def _after(
            conn: Any,
            cursor: Any,
            statement: str,
            parameters: Any,
            context: Any,
            executemany: bool,
        ) -> None:
            started = getattr(context, "_auto_ontology_started", None)
            if started is None:
                return
            elapsed = time.perf_counter() - started
            # Logged, not accumulated. A running total would need a lock on
            # every statement, and the total is not the useful number anyway:
            # per-statement duration is the only figure that is attributable.
            # A caller cannot subtract "SQL time during my window" from its own
            # wall time: the statements run on a shared event-loop thread, so
            # that window also contains every concurrent caller's queries and
            # over-counts by roughly the concurrency level.
            logger.debug("vdb.sql %.4fs", elapsed)
    except Exception:  # pragma: no cover - instrumentation must never break I/O
        logger.debug("Could not instrument VDB engine timing", exc_info=True)


def _pool_status(engine: PGEngine) -> str:
    """One-line pool occupancy, or '?' when it cannot be read."""
    try:
        target = getattr(engine, "_pool", None) or getattr(engine, "_engine", None)
        sync_engine = getattr(target, "sync_engine", None) or target
        return sync_engine.pool.status()
    except Exception:  # pragma: no cover - status is diagnostic, never load-bearing
        logger.debug("Could not read VDB pool status", exc_info=True)
        return "?"


class PostgresVDB:
    """Vector store backed by Postgres + pgvector via LangChain v2.

    Each record becomes a :class:`Document` whose ``page_content`` is
    the searchable text and whose ``metadata`` carries the original record
    metadata plus ``document_type``. Embeddings produced upstream by the NIM
    pipeline are bulk-loaded into PGVectorStore via ``add_embeddings`` so we
    don't re-run the embedder on the write path.

    Source projections use stable vector IDs and a promoted ``connection_id``
    column for scoped deletion and similarity search.
    """

    # Tells upstream ``search_semantic_index`` to build per-query metadata
    # filters as a flat ``{column: value | [values]}`` mapping instead of a
    # SQL ``LIKE`` predicate over the JSON metadata column. The dict is fed
    # straight into ``PGVectorStore.similarity_search_with_score_by_vector``.
    metadata_filter_format = "dict"

    def __init__(self, *, reset: bool = False, **kwargs: Any) -> None:
        connection_string = kwargs.get("connection_string")
        if not connection_string:
            raise ValueError(
                "PostgresVDB requires a 'connection_string' kwarg "
                "(e.g. postgresql://user:pass@host:5432/dbname)."
            )
        self.connection_string: str = connection_string

        self.collection_name: str = kwargs.get(
            "collection_name", kwargs.get("index_name", "data_objects_layer")
        )
        self.schema_name: str = kwargs.get("schema_name", "public")
        self.embeddings: Embeddings = kwargs.get("embeddings") or _UnusableEmbeddings()

        self._engine: Optional[PGEngine] = None
        self._store: Optional[PGVectorStore] = None
        self.vector_size: Optional[int] = kwargs.get("vector_size")
        self.database_name = kwargs.get("database_name")
        self.global_pql: bool = kwargs.get("global_pql", False)
        self.connection_id = kwargs.get("connection_id")
        if self.global_pql and self.connection_id is not None:
            raise ValueError(
                "invalid_global_pql: a global PQL client cannot bind a source"
            )
        if self.connection_id is not None:
            self._source_scope(self.connection_id)
        if reset:
            self.delete_by_connection(self._source_scope(self.connection_id))

    def _source_scope(self, connection_id: Any) -> str:
        """Validate source selection before any store or embedding I/O."""
        if self.global_pql:
            raise ValueError("invalid_global_pql: global PQL cannot select a source")
        if not isinstance(connection_id, str) or not connection_id.strip():
            raise ValueError("source_required: a single connection_id is required")
        if self.connection_id is not None and connection_id != self.connection_id:
            raise ValueError("source_conflict: connection_id differs from bound source")
        return connection_id

    def _require_source_schema(self) -> None:
        """Existing collections must be explicitly migrated, never backfilled here."""
        with psycopg.connect(self.connection_string) as conn:
            row = conn.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = %s AND column_name = 'connection_id' AND data_type = 'text'",
                (self.schema_name, self.collection_name),
            ).fetchone()
        if row is None:
            raise ValueError(
                "source_not_ready: vector collection requires identity migration"
            )

    # ------------------------------------------------------------------
    # Engine / store lifecycle
    # ------------------------------------------------------------------

    def _get_engine(self) -> PGEngine:
        if self._engine is None:
            # from_connection_string forwards **kwargs to create_async_engine.
            # Passing none leaves SQLAlchemy's default pool_size=5, the ceiling
            # every concurrent retrieval shares: at 12 concurrent questions a
            # small share of retrievals found it full. Raising it did not move
            # wall clock -- the queries were not what those callers were waiting
            # on -- so the default is set to clear the ceiling, not for headroom
            # nothing used.
            #
            # Budget connections before raising these. Auto Ontology opens two of these
            # stores (data and semantic), each able to reach
            # pool_size + max_overflow, and the catalog pool in auto_ontology.dal.session
            # is separate and not covered by AUTO_ONTOLOGY_PG_POOL_SIZE.
            self._engine = PGEngine.from_connection_string(
                _to_async_url(self.connection_string),
                pool_size=int(os.environ.get("AUTO_ONTOLOGY_VDB_POOL_SIZE", "10")),
                max_overflow=int(
                    os.environ.get("AUTO_ONTOLOGY_VDB_MAX_OVERFLOW", "10")
                ),
                pool_pre_ping=True,
            )
            _instrument_engine(self._engine)
        return self._engine

    def _table_exists(self) -> bool:
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (self.schema_name, self.collection_name),
                )
                return cur.fetchone() is not None

    def _ensure_schema(self) -> None:
        """Create the target schema if it doesn't exist.

        ``init_vectorstore_table`` creates the table (and the pgvector
        extension) but not its containing schema, so it must exist first.
        """
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name` is internal config (defaults to 'public');
                # psycopg can't parameterise identifiers, so it's interpolated
                # via the identifier-safe quote. Not user input.
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema_name)
                    )
                )
            conn.commit()

    def _get_store(self, *, create: bool = False) -> Optional[PGVectorStore]:
        """Return the vector store, creating the table on first write.

        Missing collections return ``None`` unless explicitly creating with a
        known dimension. Existing legacy schemas require operator migration.
        """
        if self._store is not None:
            return self._store

        if not self._table_exists():
            if not create:
                return None
            if not self.vector_size:
                raise ValueError("vector_size is required to create a collection")
            engine = self._get_engine()
            self._ensure_schema()
            engine.init_vectorstore_table(
                table_name=self.collection_name,
                vector_size=self.vector_size,
                schema_name=self.schema_name,
                metadata_columns=[
                    Column(_DATABASE_METADATA_COLUMN, "VARCHAR(100)", nullable=True),
                    Column(_LABEL_METADATA_COLUMN, "VARCHAR(100)", nullable=True),
                    Column(_CONNECTION_METADATA_COLUMN, "TEXT", nullable=True),
                ],
            )
            with psycopg.connect(self.connection_string) as conn:
                conn.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} (connection_id)"
                    ).format(
                        sql.Identifier(f"ix_{self.collection_name}_connection_id"),
                        sql.Identifier(self.schema_name, self.collection_name),
                    )
                )
            self._create_vector_index()

        self._require_source_schema()
        self._store = PGVectorStore.create_sync(
            engine=self._get_engine(),
            embedding_service=self.embeddings,
            table_name=self.collection_name,
            schema_name=self.schema_name,
            metadata_columns=[
                _DATABASE_METADATA_COLUMN,
                _LABEL_METADATA_COLUMN,
                _CONNECTION_METADATA_COLUMN,
            ],
        )
        return self._store

    def _create_vector_index(self) -> None:
        """Convert the embedding column to ``halfvec`` and index it with HNSW.

        Without an index every similarity search is a sequential scan over the
        whole collection -- measured at 175ms against 0.36ms indexed on 51,200
        rows, and it degrades linearly from there.

        **The column has to become ``halfvec`` for the index to be reachable**,
        which is not obvious and is worth the ALTER. pgvector caps HNSW at 2000
        dimensions for ``vector`` and the embedding model emits 2048, so the
        only index that can be built over a ``vector(2048)`` column is one over
        a cast expression -- and an expression index is matched *syntactically*,
        so it serves ``embedding::halfvec(2048) <=> $1`` and not the plain
        ``embedding <=> $1`` that ``PGVectorStore`` generates. Postgres then
        silently plans a sequential scan and nothing reports it. Typing the
        column ``halfvec`` instead lets a plain index serve the query the store
        actually writes: the *parameter* is cast (``embedding <=> ($1)::halfvec``),
        which costs nothing, rather than the indexed column.

        The tradeoff is that stored vectors are float16 rather than float32.
        Cosine ranking over 2048 dimensions tolerates that -- it is the same
        approximation the index itself would apply -- but it is a real precision
        change, not just an index.

        Best-effort: a failure here costs latency, not correctness, so it is
        logged and the store still works.
        """
        qualified = sql.Identifier(self.schema_name, self.collection_name)
        index_name = sql.Identifier(f"ix_hnsw_{self.collection_name}_embedding")
        try:
            with psycopg.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} "
                            "ALTER COLUMN embedding TYPE halfvec({dim})"
                        ).format(
                            table=qualified,
                            dim=sql.Literal(self.vector_size),
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                            "USING hnsw (embedding halfvec_cosine_ops)"
                        ).format(name=index_name, table=qualified)
                    )
                conn.commit()
        except Exception:
            logger.warning(
                "PostgresVDB: could not build the HNSW index on %s.%s; "
                "similarity search will fall back to a sequential scan",
                self.schema_name,
                self.collection_name,
                exc_info=True,
            )

    def create_index(self, **kwargs: Any) -> str:
        """Ensure the pgvector extension and underlying table exist."""
        self._get_store(create=True)
        return self.collection_name

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def write_to_index(
        self,
        records: list,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> int:
        """Upsert source projections, returning the number of rows written."""
        documents: list[Document] = []
        embeddings: list[list[float]] = []
        ids: list[str] = []
        skipped = 0

        for record in _flatten(records):
            metadata = record.get("metadata") or {}
            embedding = metadata.get("embedding")
            text = (
                record.get("text") or record.get("content") or metadata.get("content")
            )
            if not embedding:
                skipped += 1
                continue

            if self.global_pql:
                validate_global_pql_metadata(metadata)
                ids.append(
                    global_pql_projection_id(self.collection_name, metadata["id"])
                )
            else:
                if metadata.get("label") == "PqlAnalysis":
                    raise ValueError(
                        "invalid_global_pql: use the explicit global PQL client"
                    )
                cid = self._source_scope(metadata.get("connection_id"))
                ids.append(projection_id(self.collection_name, cid, metadata.get("id")))

            doc_metadata = {
                "document_type": record.get("document_type"),
                **{k: v for k, v in metadata.items() if k != "embedding"},
            }
            documents.append(Document(page_content=text, metadata=doc_metadata))
            embeddings.append([float(v) for v in embedding])

        if not documents:
            logger.info(
                "PostgresVDB.write_to_index: no rows to insert (skipped %d)", skipped
            )
            return 0

        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1 or (
            self.vector_size is not None and self.vector_size not in dimensions
        ):
            raise ValueError("embedding dimensions do not match vector_size")
        self.vector_size = len(embeddings[0])
        store = self._get_store(create=True)
        assert store is not None  # vector_size was provided

        inserted = 0
        for start in range(0, len(documents), batch_size):
            chunk_docs = documents[start : start + batch_size]
            chunk_embs = embeddings[start : start + batch_size]
            store.add_embeddings(
                texts=[d.page_content for d in chunk_docs],
                embeddings=chunk_embs,
                metadatas=[d.metadata for d in chunk_docs],
                ids=ids[start : start + batch_size],
            )
            inserted += len(chunk_docs)

        logger.info(
            "PostgresVDB.write_to_index: inserted %d rows into %s (skipped %d)",
            inserted,
            self.collection_name,
            skipped,
        )
        return inserted

    def delete_by_database(self, database_name: str) -> list[str]:
        """Reject unsafe legacy physical-name deletion without touching storage."""
        raise ValueError("source_required: delete by connection_id, not database_name")

    def delete_by_connection(self, connection_id: str) -> list[str]:
        """Atomically delete just the selected source and return vector IDs."""
        connection_id = self._source_scope(connection_id)
        if not self._table_exists():
            return []
        self._require_source_schema()
        with psycopg.connect(self.connection_string) as conn:
            rows = conn.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE connection_id = %s RETURNING langchain_id"
                ).format(sql.Identifier(self.schema_name, self.collection_name)),
                (connection_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def projection_connection_ids(self, node_id: str) -> list[str]:
        """Read only promoted source IDs for vocabulary refresh; no model calls."""
        if self.global_pql:
            raise ValueError(
                "invalid_global_pql: global vocabulary has no source projections"
            )
        if not self._table_exists():
            return []
        self._require_source_schema()
        with psycopg.connect(self.connection_string) as conn:
            rows = conn.execute(
                sql.SQL(
                    "SELECT DISTINCT connection_id FROM {} WHERE langchain_metadata ->> 'id' = %s AND connection_id IS NOT NULL ORDER BY connection_id"
                ).format(sql.Identifier(self.schema_name, self.collection_name)),
                (node_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def delete_all(self) -> int:
        """Delete the bound source, or every row for an unbound administrative client.

        Returns the number of rows deleted (``0`` when the collection table
        doesn't exist yet).
        """
        if self.global_pql:
            return self._delete_global_pql()
        if self.connection_id is not None:
            return len(self.delete_by_connection(self.connection_id))
        if not self._table_exists():
            logger.info(
                "PostgresVDB.delete_all: collection %s not found, nothing to delete",
                self.collection_name,
            )
            return 0

        self._require_source_schema()
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name`/`collection_name` are internal config; psycopg
                # can't parameterise identifiers, so they're composed via the
                # identifier-safe API. Not user input.
                cur.execute(
                    sql.SQL("DELETE FROM {table}").format(
                        table=sql.Identifier(self.schema_name, self.collection_name)
                    )
                )
                deleted = cur.rowcount

        logger.info(
            "PostgresVDB.delete_all: deleted %d rows from %s",
            deleted,
            self.collection_name,
        )
        return deleted

    def _delete_global_pql(self, node_id: str | None = None) -> int:
        """Global deletes never match source projections or other semantic labels."""
        if not self._table_exists():
            return 0
        self._require_source_schema()
        predicate = sql.SQL("label = 'PqlAnalysis' AND connection_id IS NULL")
        params: tuple = ()
        if node_id is not None:
            predicate += sql.SQL(" AND langchain_metadata ->> 'id' = %s")
            params = (node_id,)
        with psycopg.connect(self.connection_string) as conn:
            result = conn.execute(
                sql.SQL("DELETE FROM {} WHERE {}").format(
                    sql.Identifier(self.schema_name, self.collection_name), predicate
                ),
                params,
            )
            return result.rowcount

    def delete_by_id(self, node_id: str, *, connection_id: str | None = None) -> int:
        """Delete one source projection, never all occurrences of a node."""
        if self.global_pql:
            if connection_id is not None:
                raise ValueError(
                    "invalid_global_pql: global delete cannot select a source"
                )
            global_pql_projection_id(self.collection_name, node_id)
            return self._delete_global_pql(node_id)
        if connection_id is None:
            raise TypeError(
                "delete_by_id requires keyword-only connection_id for source projections"
            )
        connection_id = self._source_scope(connection_id)
        vector_id = projection_id(self.collection_name, connection_id, node_id)
        if not self._table_exists():
            logger.info(
                "PostgresVDB.delete_by_id: collection %s not found, nothing to delete",
                self.collection_name,
            )
            return 0

        self._require_source_schema()
        with psycopg.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # `schema_name`/`collection_name` are internal config; psycopg
                # can't parameterise identifiers, so they're composed via the
                # identifier-safe API. Not user input.
                cur.execute(
                    sql.SQL(
                        """
                        DELETE FROM {table}
                        WHERE langchain_id = %s AND connection_id = %s
                        """
                    ).format(
                        table=sql.Identifier(self.schema_name, self.collection_name)
                    ),
                    (vector_id, connection_id),
                )
                deleted = cur.rowcount

        logger.info(
            "PostgresVDB.delete_by_id: deleted %d rows from %s for id=%s",
            deleted,
            self.collection_name,
            node_id,
        )
        return deleted

    def retrieval(
        self,
        queries: list,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[dict]]:
        """Cosine-similarity k-NN search for each query vector.

        ``kwargs["where"]`` is a langchain-postgres filter dict
        (e.g. ``{"label": "CustomAnalysis", "connection_id": "source-id"}``)
        applied to the declared metadata columns; passed straight through to
        ``PGVectorStore.similarity_search_with_score_by_vector``.

        A single connection ID is required, either bound or in ``where``.
        """
        where = kwargs.get("where")
        if where is not None and not isinstance(where, dict):
            raise ValueError(
                "source_required: where must select a single connection_id"
            )
        where = dict(where or {})
        if self.global_pql:
            if (
                where.get("label", "PqlAnalysis") != "PqlAnalysis"
                or "connection_id" in where
                or any(key.startswith("$") for key in where)
            ):
                raise ValueError(
                    "invalid_global_pql: filters must select only global PqlAnalysis"
                )
            where.update(label="PqlAnalysis", connection_id={"$exists": False})
        else:
            where["connection_id"] = self._source_scope(
                where.get("connection_id", self.connection_id)
            )
        store = self._get_store()
        if store is None:
            return [[] for _ in queries]

        # Wall time for this call. Server-side time is deliberately not
        # summarised here: a global SQL accumulator sampled around this window
        # also captures every concurrent caller's statements, so any per-call
        # figure derived that way over-reports by roughly the concurrency
        # level. The attributable number is the per-statement `vdb.sql` line.
        _started = time.perf_counter()

        results: list[list[dict]] = []
        for query in queries:
            try:
                hits = store.similarity_search_with_score_by_vector(
                    embedding=query,
                    filter=where,
                    k=top_k,
                )
                results.append(
                    [
                        {
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "_distance": float(score),
                        }
                        for doc, score in hits
                    ]
                )
            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
                return [[] for _ in queries]

        logger.debug(
            "vdb.retrieval total=%.3fs queries=%d pool=%s",
            time.perf_counter() - _started,
            len(queries),
            _pool_status(self._get_engine()),
        )
        return results

    def run(self, records: list) -> int:
        """Create the collection if needed, then write records to it."""
        return self.write_to_index(records)

    #: Seconds to wait for the pool to dispose before giving up on it.
    #:
    #: Disposal is a courtesy: the pool's sockets are closed by the OS when the
    #: process exits either way. Waiting longer than this buys nothing and risks
    #: the hang described on :meth:`close`.
    _DISPOSE_TIMEOUT = 5.0

    def close(self) -> None:
        """Dispose the engine's connection pool, without ever blocking forever.

        Three paths, because disposal has to cross a thread boundary and each
        way of doing that can wedge:

        * **On the engine's own loop thread** -- scheduled, never waited on, or
          it would join itself.
        * **Off it, with the loop alive** -- submitted and waited on with a
          timeout.
        * **Off it, with the loop gone** -- skipped entirely.

        That last case is the one that matters, and it is why this does not use
        ``engine._run_as_sync``: that helper waits on ``future.result()`` with
        no timeout, and at interpreter shutdown the loop thread is already gone,
        so the future is never resolved by anyone. Reached through
        :meth:`__del__`, it hung ``python -m auto_ontology.semantic`` for 45+ minutes
        after the work had finished and committed, and the ``try/except`` around
        it cannot catch a hang.
        """
        self._store = None
        engine = self._engine
        if engine is None:
            return
        self._engine = None

        loop = getattr(engine, "_loop", None)
        thread = getattr(engine, "_thread", None)

        on_loop_thread = thread is not None and thread is threading.current_thread()
        if not on_loop_thread:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            on_loop_thread = running is not None and running is loop

        if on_loop_thread:
            # Can't block on the loop from within it; schedule dispose
            # non-blocking so it runs once the loop is free again.
            if loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(engine._pool.dispose(), loop)
                except Exception:
                    pass
            return

        # Nobody is left to run the coroutine: submitting it would wait on a
        # future that can never resolve.
        if loop is None or not loop.is_running():
            return
        if thread is not None and not thread.is_alive():
            return

        try:
            future = asyncio.run_coroutine_threadsafe(engine._pool.dispose(), loop)
            future.result(timeout=self._DISPOSE_TIMEOUT)
        except Exception:
            # Includes the timeout: the pool outliving the process by a few
            # seconds is not worth blocking a shutdown over.
            logger.debug("PostgresVDB.close: pool dispose did not complete")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
