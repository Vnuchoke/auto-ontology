# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge ReAct catalog additions and stable connection identity.

Revision ID: f4a39d8b2160
Revises: caee572fa81c, e38c06a17f42
"""

revision: str = "f4a39d8b2160"
down_revision: tuple[str, str] = ("caee572fa81c", "e38c06a17f42")
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Both parent branches already perform their independent schema changes."""


def downgrade() -> None:
    """Leave each parent's rollback and backfill safeguards intact."""
