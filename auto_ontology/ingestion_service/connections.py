# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata-only, env-first source discovery for both schedulers."""

from auto_ontology.connectors.sources import list_active_sources


def resolve_connection_ids() -> list[str]:
    """Return active IDs in configured order, without opening any connector."""
    return [source.connection_id for source in list_active_sources()]
