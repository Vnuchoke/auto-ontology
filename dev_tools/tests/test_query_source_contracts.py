# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query callers must not silently fall back after a missing CID keyword."""

from collections.abc import Callable
from inspect import signature
from typing import Any

import pytest

from auto_ontology.dal import attributes, exploration


@pytest.mark.parametrize(
    ("helper", "argument"),
    [
        (attributes.find_connected_junction_tables, ["table"]),
        (attributes.find_anchor_hub_siblings, "table"),
        (attributes.column_participates_in_semantic_fk, "column"),
        (exploration.fetch_column_exploration_details, "column"),
        (exploration.fetch_column_attribute_exploration_details, "attribute"),
        (exploration.fetch_sql_attribute_exploration_details, "sql-attribute"),
    ],
)
def test_retrieval_helpers_accept_the_source_keyword_used_by_their_callers(
    helper: Callable[..., Any], argument: object
) -> None:
    # Binding checks the real interface without opening a store or masking TypeError.
    signature(helper).bind(argument, connection_id="selected-source")
