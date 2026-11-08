# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit source identities for isolated graph tests, never runtime defaults."""

SOURCE = {
    "connection_id": "test-source",
    "display_label": "Test source",
    "database_name": "test_db",
}


def graph_state(payload: dict) -> dict:
    return {
        "connection_id": SOURCE["connection_id"],
        "source": dict(SOURCE),
        "path_state": {
            "connection_id": SOURCE["connection_id"],
            "target_db": SOURCE["database_name"],
        },
    }
