# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock

import pytest

from auto_ontology.connectors import sources


@pytest.fixture
def active(monkeypatch: pytest.MonkeyPatch) -> list[sources.SourceDescriptor]:
    rows = [
        sources.SourceDescriptor(
            "opaque-A", "Sales East", "sales", "postgres", "managed"
        ),
        sources.SourceDescriptor(
            "opaque-B", "Sales West", "sales", "postgres", "managed"
        ),
    ]
    monkeypatch.setattr(sources, "list_active_sources", lambda: rows)
    monkeypatch.setattr(
        sources.connections,
        "normalize_source_labels",
        lambda labels: [s.strip().lower() for s in labels],
    )
    return rows


@pytest.mark.parametrize(
    "kwargs,status,code",
    [
        ({}, 422, "source_required"),
        ({"connection_id": " "}, 422, "invalid_source_selector"),
        ({"connection_id": 1}, 422, "invalid_source_selector"),
        (
            {"connection_id": "opaque-A", "target_db": "sales"},
            422,
            "invalid_source_selector",
        ),
        ({"target_db": "SALES"}, 409, "ambiguous_source"),
        ({"connection_id": "missing"}, 404, "source_unavailable"),
        ({"connection_label": "missing"}, 404, "source_unavailable"),
        ({"target_db": "Sales East"}, 404, "source_unavailable"),
    ],
)
def test_selection_errors(
    active: list[sources.SourceDescriptor],
    kwargs: dict[str, Any],
    status: int,
    code: str,
) -> None:
    with pytest.raises(sources.SourceSelectionError) as exc:
        sources.resolve_source(**kwargs)
    assert (exc.value.status_code, exc.value.code, str(exc.value)) == (
        status,
        code,
        code,
    )


def test_explicit_id_and_postgres_normalized_label(
    active: list[sources.SourceDescriptor], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert sources.resolve_source(connection_id="opaque-A") == active[0]
    normalize = Mock(return_value=["same", "same", "other"])
    monkeypatch.setattr(sources.connections, "normalize_source_labels", normalize)
    assert sources.resolve_source(connection_label="PG-specific") == active[0]
    normalize.assert_called_once_with(["PG-specific", "Sales East", "Sales West"])


def test_zero_one_and_legacy_uuid(active: list[sources.SourceDescriptor]) -> None:
    active.pop()
    assert sources.resolve_source() == active[0]
    assert sources.resolve_source(target_db="SALES") == active[0]
    active[0] = sources.SourceDescriptor(
        "42d62c22-b1c4-4ed7-8173-6aec9e2bd432", "East", "sales", "postgres", "managed"
    )
    assert sources.resolve_source(target_db=active[0].connection_id) == active[0]
    active.clear()
    with pytest.raises(sources.SourceSelectionError) as exc:
        sources.resolve_source()
    assert (exc.value.status_code, exc.value.code) == (503, "source_unavailable")


def test_metadata_failure_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sources, "list_active_sources", Mock(side_effect=RuntimeError("secret-url"))
    )
    with pytest.raises(sources.SourceSelectionError) as exc:
        sources.resolve_source(connection_id="A")
    assert (exc.value.status_code, str(exc.value)) == (503, "source_not_ready")


def test_invalid_selector_does_not_read_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = Mock(side_effect=AssertionError("metadata called"))
    monkeypatch.setattr(sources, "list_active_sources", metadata)
    with pytest.raises(sources.SourceSelectionError):
        sources.resolve_source(connection_id="A", connection_label="A")
    metadata.assert_not_called()
