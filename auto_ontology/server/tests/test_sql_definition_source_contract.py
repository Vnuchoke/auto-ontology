# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.server.models import CustomAnalysis, SqlAttribute


@pytest.mark.parametrize("model", [CustomAnalysis, SqlAttribute])
def test_definition_schema_exposes_nullable_legacy_source(model: type) -> None:
    schema = model.model_json_schema()
    assert "connection_id" in schema["properties"]
    assert "connection_id" not in schema["required"]
    assert model(id="legacy").connection_id is None
    assert (
        model(id="owned", connection_id="source").model_dump()["connection_id"]
        == "source"
    )
