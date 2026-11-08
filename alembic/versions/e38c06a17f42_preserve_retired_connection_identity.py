# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve retired connection identities after catalog removal."""

from alembic import op
import sqlalchemy as sa

revision: str = "e38c06a17f42"
down_revision: str | None = "d27b94f58c31"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retired_connection",
        sa.Column("connection_id", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("connection_id", name="pk_retired_connection"),
    )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(sa.text("SELECT EXISTS (SELECT 1 FROM retired_connection)"))
        .scalar_one()
    ):
        raise RuntimeError(
            "Retired connection identities exist; restore a coordinated backup instead"
        )
    op.drop_table("retired_connection")
