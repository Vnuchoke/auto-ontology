# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.server.models import CustomAnalysis, SqlAttribute
from auto_ontology.server.sql_attributes.router import (
    SqlAttributeCreate,
    SqlAttributeUpdate,
    SqlAttributeValidate,
)


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


@pytest.mark.parametrize(
    "model", [SqlAttributeValidate, SqlAttributeCreate, SqlAttributeUpdate]
)
def test_sql_definition_inputs_keep_source_and_full_query_contract(model: type) -> None:
    properties = model.model_json_schema()["properties"]
    assert "connection_id" in properties
    assert "full SQL query" in properties["expression"]["description"]
