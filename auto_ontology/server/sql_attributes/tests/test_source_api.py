# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The API must not discard source selectors before service validation."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auto_ontology.server.sql_attributes import router as attrs
from auto_ontology.server.datasources import router as analyses


@pytest.mark.parametrize(
    "module, model, handler, method, payload",
    [
        (
            attrs,
            attrs.SqlAttributeValidate,
            attrs.validate_sql_attribute,
            "validate_sql_attribute",
            {"expression": "SELECT 1"},
        ),
        (
            analyses,
            analyses.CustomAnalysisValidate,
            analyses.validate_custom_analysis_sql,
            "validate_custom_analysis_sql",
            {"sql": "SELECT 1"},
        ),
    ],
)
def test_validate_preserves_selector_and_returns_422(
    monkeypatch, module, model, handler, method, payload
):
    service = module.dal if module is attrs else module.custom_analyses_dal
    validate = Mock(side_effect=ValueError("source_required"))
    monkeypatch.setattr(service, method, validate)
    body = model(**payload, connection_id="B")
    with pytest.raises(HTTPException) as error:
        handler(body)
    assert error.value.status_code == 422
    assert validate.call_args.kwargs["connection_id"] == "B"


@pytest.mark.parametrize(
    "model",
    [
        attrs.SqlAttributeCreate,
        attrs.SqlAttributeUpdate,
        attrs.SqlAttributeMetadataPatch,
        analyses.CustomAnalysisCreate,
    ],
)
def test_write_body_keeps_connection_id(model):
    body = model(
        name="Orders",
        description="Count",
        expression="SELECT 1",
        sql="SELECT 1",
        term_id="term",
        connection_id="B",
    )
    assert body.model_dump()["connection_id"] == "B"


@pytest.mark.parametrize("operation", ["create", "update", "patch"])
def test_attribute_writes_forward_selector(monkeypatch, operation):
    monkeypatch.setattr(attrs, "edit_subject", Mock(return_value=None))
    monkeypatch.setattr(attrs, "owner_from_body", Mock(return_value=None))
    monkeypatch.setattr(attrs, "resolve_internal_user", Mock(return_value=None))
    method = "create_sql_attribute" if operation == "create" else "update_sql_attribute"
    write = Mock(side_effect=ValueError("source_conflict"))
    monkeypatch.setattr(attrs.dal, method, write)
    model = {
        "create": attrs.SqlAttributeCreate,
        "update": attrs.SqlAttributeUpdate,
        "patch": attrs.SqlAttributeMetadataPatch,
    }[operation]
    body = model(
        name="Orders",
        description="Count",
        expression="SELECT 1",
        term_id="term",
        connection_id="B",
    )
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as error:
        if operation == "create":
            attrs.create_sql_attribute(request, body)
        elif operation == "update":
            attrs.update_sql_attribute(request, "definition", body)
        else:
            attrs.patch_sql_attribute(request, "definition", body)
    assert error.value.status_code == 422
    assert write.call_args.kwargs["connection_id"] == "B"


@pytest.mark.parametrize("operation", ["create", "update"])
def test_analysis_writes_forward_selector(monkeypatch, operation):
    method = f"{operation}_custom_analysis"
    write = Mock(side_effect=ValueError("source_conflict"))
    monkeypatch.setattr(analyses.custom_analyses_dal, method, write)
    body = analyses.CustomAnalysisCreate(
        name="Orders", description="Count", sql="SELECT 1", connection_id="B"
    )
    with pytest.raises(HTTPException) as error:
        if operation == "create":
            analyses.create_custom_analysis(body)
        else:
            analyses.update_custom_analysis("definition", body)
    assert error.value.status_code == 422
    assert write.call_args.kwargs["connection_id"] == "B"
