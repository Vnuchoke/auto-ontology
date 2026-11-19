# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize the Concepts revision shipped on master.

On other branches this is a marker only: running master's original DDL
would discard additional semantic FK targets before they can be migrated.
"""

from alembic import op

revision = "0b6f3c2a9d14"
down_revision = "caee572fa81c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_context().environment_context.get_revision_argument() == revision:
        raise RuntimeError(
            "The master marker is not an upgrade target; upgrade to head"
        )


def downgrade() -> None:
    pass
