# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add source-scoped Concepts without dropping legacy semantic FK targets.

Revision ID: 38a47b1ef206
Revises: f5b40e9c3271
Create Date: 2026-11-18 12:00:00.000000

Every Term gets a one-member Concept per represented source, and every
semantic FK is re-pointed from the ColumnAttribute it targeted to a key on that
attribute's Concept, held by the attribute's own column. Joins after the
upgrade therefore land on exactly the columns they landed on before it.

The unused ``join_path_edge`` view is dropped: it read
``column__semantic_fk.attribute_id``, which this revision removes.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "38a47b1ef206"
down_revision: str | Sequence[str] | None = "f5b40e9c3271"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = sa.text("(gen_random_uuid())::text")

_JOIN_PATH_EDGE_VIEW_SQL = """
CREATE OR REPLACE VIEW join_path_edge AS
    SELECT 'column'::text AS src_kind, c.id       AS src_id,
           'table'::text  AS dst_kind, c.table_id AS dst_id
      FROM catalog_column c
    UNION ALL
    SELECT 'table', c.table_id, 'column', c.id
      FROM catalog_column c
    UNION ALL
    SELECT 'column', h.column_id, 'column_attribute', h.attribute_id
      FROM column__has_attribute h
    UNION ALL
    SELECT 'column_attribute', h.attribute_id, 'column', h.column_id
      FROM column__has_attribute h
    UNION ALL
    SELECT 'column', f.column_id, 'column_attribute', f.attribute_id
      FROM column__semantic_fk f
"""


def _create_tables() -> None:
    op.create_table(
        "concept",
        sa.Column("id", sa.Text(), server_default=_UUID, nullable=False),
        sa.Column("database_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "source", sa.Text(), server_default=sa.text("'semantic'"), nullable=False
        ),
        sa.Column(
            "certified", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("modified", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=True),
        sa.Column("modified_by", sa.Text(), nullable=True),
        sa.Column("imported_id", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["database_id"],
            ["catalog_database.id"],
            name=op.f("fk_concept_database_id_catalog_database"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept")),
        sa.UniqueConstraint("database_id", "name", name="uq_concept_database_name"),
    )
    op.create_index("ix_concept_database_id", "concept", ["database_id"])
    op.create_index("ix_concept_imported_id", "concept", ["imported_id"])
    op.create_table(
        "term__concept",
        sa.Column("term_id", sa.Text(), nullable=False),
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.Column("database_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["database_id"],
            ["catalog_database.id"],
            name=op.f("fk_term__concept_database_id_catalog_database"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_term__concept_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["term_id"],
            ["term.id"],
            name=op.f("fk_term__concept_term_id_term"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("term_id", "concept_id", name=op.f("pk_term__concept")),
        sa.UniqueConstraint(
            "term_id", "database_id", name=op.f("uq_term__concept_term_id")
        ),
    )
    op.create_index("ix_term__concept_concept_id", "term__concept", ["concept_id"])
    op.create_table(
        "concept_key",
        sa.Column("id", sa.Text(), server_default=_UUID, nullable=False),
        sa.Column("concept_id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["concept_id"],
            ["concept.id"],
            name=op.f("fk_concept_key_concept_id_concept"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_concept_key")),
    )
    op.create_index("ix_concept_key_concept_id", "concept_key", ["concept_id"])
    op.create_table(
        "concept_key__column",
        sa.Column("key_id", sa.Text(), nullable=False),
        sa.Column("column_id", sa.Text(), nullable=False),
        sa.Column("table_id", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["catalog_column.id"],
            name=op.f("fk_concept_key__column_column_id_catalog_column"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["key_id"],
            ["concept_key.id"],
            name=op.f("fk_concept_key__column_key_id_concept_key"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["catalog_table.id"],
            name=op.f("fk_concept_key__column_table_id_catalog_table"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "key_id", "column_id", name=op.f("pk_concept_key__column")
        ),
        sa.UniqueConstraint("column_id", name=op.f("uq_concept_key__column_column_id")),
        sa.UniqueConstraint(
            "key_id", "table_id", name=op.f("uq_concept_key__column_key_id")
        ),
    )
    op.create_index(
        "ix_concept_key__column_table_id", "concept_key__column", ["table_id"]
    )


def _backfill_concepts() -> None:
    op.execute(
        """
        CREATE TEMPORARY TABLE _concept_term ON COMMIT DROP AS
        WITH term_db AS (
            SELECT DISTINCT
                   tt.term_id, s.database_id
              FROM table__term tt
              JOIN catalog_table t ON t.id = tt.table_id
              JOIN catalog_schema s ON s.id = t.schema_id
        ), named AS (
            SELECT term.id AS term_id, term_db.database_id, term.name, term.description,
                   row_number() OVER (
                       PARTITION BY term_db.database_id, term.name
                       ORDER BY term.id
                   ) AS rank
              FROM term JOIN term_db ON term_db.term_id = term.id
        )
        SELECT gen_random_uuid()::text AS id, term_id, database_id,
               CASE WHEN rank = 1 AND NOT EXISTS (
                    SELECT 1 FROM concept c WHERE c.database_id = named.database_id
                    AND c.name = named.name
               ) THEN name ELSE name || ' (' || term_id || ')' END AS name,
               description
          FROM named
         WHERE NOT EXISTS (
            SELECT 1 FROM term__concept tc
            WHERE tc.term_id = named.term_id AND tc.database_id = named.database_id
         )
        """
    )
    op.execute(
        "INSERT INTO concept (id, database_id, name, description) "
        "SELECT id, database_id, name, description FROM _concept_term"
    )
    op.execute(
        "INSERT INTO term__concept (term_id, concept_id, database_id) "
        "SELECT term_id, id, database_id FROM _concept_term"
    )


def _backfill_keys() -> None:
    # One key per column that a semantic FK lands on today, plus one per
    # single-column declared primary key that none lands on.
    op.execute(
        """
        CREATE TEMPORARY TABLE _concept_key_column ON COMMIT DROP AS
        WITH targets AS (
            SELECT DISTINCT target_column_id AS column_id FROM _legacy_semantic_fk
        ), key_columns AS (
            SELECT column_id FROM targets
            UNION
            SELECT c.id
              FROM catalog_table t
              JOIN catalog_column c ON c.table_id = t.id
             WHERE cardinality(t.pk) = 1 AND c.name = t.pk[1]
        )
        SELECT DISTINCT ON (c.id) (gen_random_uuid())::text AS key_id, c.id AS column_id,
               c.table_id, c.name, tc.concept_id
          FROM key_columns k
          JOIN catalog_column c ON c.id = k.column_id
          JOIN catalog_table t ON t.id = c.table_id
          JOIN catalog_schema s ON s.id = t.schema_id
          JOIN table__term tt ON tt.table_id = c.table_id
          JOIN term__concept tc ON tc.term_id = tt.term_id AND tc.database_id = s.database_id
         ORDER BY c.id, tc.concept_id
        """
    )
    op.execute(
        """
        INSERT INTO concept_key (id, concept_id, name)
        SELECT key_id, concept_id, name FROM _concept_key_column
        """
    )
    op.execute(
        """
        INSERT INTO concept_key__column (key_id, column_id, table_id)
        SELECT key_id, column_id, table_id FROM _concept_key_column
        """
    )


def _check_legacy_semantic_fks() -> None:
    # Hold the rows stable between validation and the schema rewrite.
    op.execute("LOCK TABLE column__semantic_fk IN ACCESS EXCLUSIVE MODE")
    op.execute(
        """
        CREATE TEMPORARY TABLE _legacy_semantic_fk ON COMMIT DROP AS
        SELECT f.column_id, f.attribute_id, target.id AS target_column_id,
               CASE
                 WHEN target.id IS NULL THEN 'no canonical owner column'
                 WHEN src_schema.database_id <> dst_schema.database_id THEN 'cross-source target'
                 WHEN NOT EXISTS (SELECT 1 FROM table__term tt WHERE tt.table_id = dst.id)
                      THEN 'target table has no Term'
                 WHEN count(*) OVER (PARTITION BY f.column_id, target.id) > 1
                      THEN 'targets would collapse onto one key'
               END AS reason
          FROM column__semantic_fk f
          JOIN catalog_column src_col ON src_col.id = f.column_id
          JOIN catalog_table src ON src.id = src_col.table_id
          JOIN catalog_schema src_schema ON src_schema.id = src.schema_id
          JOIN column_attribute a ON a.id = f.attribute_id
          LEFT JOIN catalog_table dst ON dst.id = a.table_id
          LEFT JOIN catalog_schema dst_schema ON dst_schema.id = dst.schema_id
          LEFT JOIN catalog_column target ON target.table_id = a.table_id
                 AND target.name = a.source_column
                 AND EXISTS (SELECT 1 FROM column__has_attribute h
                              WHERE h.attribute_id = a.id AND h.column_id = target.id)
        """
    )
    op.execute(
        """
        DO $$
        DECLARE failures text;
        BEGIN
            SELECT string_agg(format('%s -> %s (%s)', column_id, attribute_id, reason), '; ')
              INTO failures
              FROM (SELECT * FROM _legacy_semantic_fk WHERE reason IS NOT NULL
                    ORDER BY column_id, attribute_id LIMIT 10) examples;
            IF failures IS NOT NULL THEN
                RAISE EXCEPTION 'Cannot migrate semantic FKs; no edges changed. First 10: %', failures;
            END IF;
        END $$
        """
    )


def _repoint_semantic_fks() -> None:
    op.add_column("column__semantic_fk", sa.Column("key_id", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE column__semantic_fk f
           SET key_id = kc.key_id
          FROM _legacy_semantic_fk legacy
          JOIN concept_key__column kc ON kc.column_id = legacy.target_column_id
         WHERE f.column_id = legacy.column_id AND f.attribute_id = legacy.attribute_id
        """
    )
    op.drop_index("ix_column__semantic_fk_attribute_id", "column__semantic_fk")
    op.drop_constraint(
        "fk_column__semantic_fk_attribute_id_column_attribute",
        "column__semantic_fk",
        type_="foreignkey",
    )
    op.drop_constraint("pk_column__semantic_fk", "column__semantic_fk", type_="primary")
    op.drop_column("column__semantic_fk", "attribute_id")
    op.alter_column("column__semantic_fk", "key_id", nullable=False)
    op.create_primary_key(
        "pk_column__semantic_fk", "column__semantic_fk", ["column_id", "key_id"]
    )
    op.create_foreign_key(
        op.f("fk_column__semantic_fk_key_id_concept_key"),
        "column__semantic_fk",
        "concept_key",
        ["key_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_column__semantic_fk_key_id", "column__semantic_fk", ["key_id"])


def upgrade() -> None:
    """Upgrade schema."""
    # A database coming from master already has Concepts and key-based FKs.
    if op.get_context().as_sql:
        raise RuntimeError(
            "Concept migration requires an online upgrade to inspect the existing schema"
        )
    inspector = sa.inspect(op.get_bind())
    fk_columns = {
        column["name"]
        for column in inspector.get_columns("column__semantic_fk", schema="public")
    }
    if inspector.has_table("concept", schema="public"):
        membership = {
            column["name"]
            for column in inspector.get_columns("term__concept", schema="public")
        }
        if membership != {"term_id", "concept_id"} or fk_columns != {
            "column_id",
            "key_id",
            "use_in_join_paths",
        }:
            raise RuntimeError("Unrecognized master Concepts schema; no edges changed")
        _upgrade_master_concepts()
        return
    _check_legacy_semantic_fks()
    op.execute("DROP VIEW IF EXISTS join_path_edge")
    _create_tables()
    _backfill_concepts()
    _backfill_keys()
    _repoint_semantic_fks()


def _upgrade_master_concepts() -> None:
    op.execute(
        "LOCK TABLE column__semantic_fk, term__concept, concept_key__column IN ACCESS EXCLUSIVE MODE"
    )
    op.execute(
        """
        DO $$
        DECLARE failures text;
        BEGIN
            SELECT string_agg(problem, '; ') INTO failures
              FROM (
                SELECT format('key %s, member column %s', kc.key_id, kc.column_id) AS problem
                  FROM concept_key__column kc
                  JOIN concept_key k ON k.id = kc.key_id
                  JOIN concept c ON c.id = k.concept_id
                  JOIN catalog_column col ON col.id = kc.column_id
                  JOIN catalog_table t ON t.id = col.table_id
                  JOIN catalog_schema s ON s.id = t.schema_id
                 WHERE kc.table_id <> t.id OR c.database_id <> s.database_id
                UNION ALL
                SELECT format('FK column %s, key %s', fk.column_id, fk.key_id)
                  FROM column__semantic_fk fk
                  JOIN concept_key k ON k.id = fk.key_id
                  JOIN concept c ON c.id = k.concept_id
                  JOIN catalog_column col ON col.id = fk.column_id
                  JOIN catalog_table t ON t.id = col.table_id
                  JOIN catalog_schema s ON s.id = t.schema_id
                 WHERE c.database_id <> s.database_id OR NOT EXISTS (
                     SELECT 1 FROM concept_key__column kc WHERE kc.key_id = fk.key_id
                 )
                 ORDER BY 1 LIMIT 10
              ) problems;
            IF failures IS NOT NULL THEN
                RAISE EXCEPTION 'Cannot migrate master Concepts: cross-source, invalid or missing key members. First 10: %', failures;
            END IF;
        END $$
        """
    )
    op.add_column("term__concept", sa.Column("database_id", sa.Text(), nullable=True))
    op.execute(
        "UPDATE term__concept tc SET database_id = c.database_id "
        "FROM concept c WHERE c.id = tc.concept_id"
    )
    op.alter_column(
        "term__concept", "database_id", existing_type=sa.Text(), nullable=False
    )
    op.create_foreign_key(
        op.f("fk_term__concept_database_id_catalog_database"),
        "term__concept",
        "catalog_database",
        ["database_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint("uq_term__concept_term_id", "term__concept", type_="unique")
    op.create_unique_constraint(
        "uq_term__concept_term_id", "term__concept", ["term_id", "database_id"]
    )
    _backfill_concepts()
    op.drop_constraint("pk_column__semantic_fk", "column__semantic_fk", type_="primary")
    op.create_primary_key(
        "pk_column__semantic_fk", "column__semantic_fk", ["column_id", "key_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    if op.get_context().environment_context.get_revision_argument() == "0b6f3c2a9d14":
        raise RuntimeError(
            "Downgrade to the master Concepts schema requires a coordinated backup"
        )
    op.execute("LOCK TABLE concept, column__semantic_fk IN ACCESS EXCLUSIVE MODE")
    if (
        op.get_bind()
        .execute(sa.text("SELECT EXISTS (SELECT 1 FROM concept)"))
        .scalar_one()
    ):
        raise RuntimeError(
            "Downgrade would discard Concepts; restore a coordinated backup instead"
        )
    op.add_column(
        "column__semantic_fk", sa.Column("attribute_id", sa.Text(), nullable=False)
    )
    op.drop_index("ix_column__semantic_fk_key_id", "column__semantic_fk")
    op.drop_constraint(
        "fk_column__semantic_fk_key_id_concept_key",
        "column__semantic_fk",
        type_="foreignkey",
    )
    op.drop_constraint("pk_column__semantic_fk", "column__semantic_fk", type_="primary")
    op.drop_column("column__semantic_fk", "key_id")
    op.create_primary_key(
        "pk_column__semantic_fk", "column__semantic_fk", ["column_id", "attribute_id"]
    )
    op.create_foreign_key(
        "fk_column__semantic_fk_attribute_id_column_attribute",
        "column__semantic_fk",
        "column_attribute",
        ["attribute_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_column__semantic_fk_attribute_id", "column__semantic_fk", ["attribute_id"]
    )
    op.drop_index("ix_concept_key__column_table_id", "concept_key__column")
    op.drop_table("concept_key__column")
    op.drop_index("ix_concept_key_concept_id", "concept_key")
    op.drop_table("concept_key")
    op.drop_index("ix_term__concept_concept_id", "term__concept")
    op.drop_table("term__concept")
    op.drop_index("ix_concept_imported_id", "concept")
    op.drop_index("ix_concept_database_id", "concept")
    op.drop_table("concept")
    op.execute(_JOIN_PATH_EDGE_VIEW_SQL)
