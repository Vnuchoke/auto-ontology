# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow shared global Terms to have independent families in source schemas."""

from alembic import op
import sqlalchemy as sa

revision: str = "f5b40e9c3271"
down_revision: str | None = "f4a39d8b2160"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_term__term_family_term_id", "term__term_family", type_="unique"
    )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM term__term_family "
                "GROUP BY term_id HAVING count(*) > 1)"
            )
        )
        .scalar_one()
    ):
        raise RuntimeError(
            "Terms belong to multiple families; restore a coordinated backup instead"
        )
    op.create_unique_constraint(
        "uq_term__term_family_term_id", "term__term_family", ["term_id"]
    )
