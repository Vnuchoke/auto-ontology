# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converge the master and develop Concept histories."""

from alembic import op

revision = "715d9c03a842"
down_revision = ("38a47b1ef206", "0b6f3c2a9d14")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    if op.get_context().environment_context.get_revision_argument() == "0b6f3c2a9d14":
        raise RuntimeError(
            "Downgrade to the master Concepts schema requires a coordinated backup"
        )
