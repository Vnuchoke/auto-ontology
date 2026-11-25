# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""merge conversation and connection histories

Revision ID: 2d718c2f16a4
Revises: 715d9c03a842, e7a3d5c19f42
Create Date: 2026-11-26 02:05:20.849018

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2d718c2f16a4"
down_revision: Union[str, Sequence[str], None] = ("715d9c03a842", "e7a3d5c19f42")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
