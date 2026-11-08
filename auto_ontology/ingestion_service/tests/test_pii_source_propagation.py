# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.ingestion_service import pii
from auto_ontology.pii_detection import service


def test_ingestion_wrapper_propagates_only_selected_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    propagate = Mock()
    monkeypatch.setattr(pii, "propagate_pii_to_attributes", propagate)
    pii.run_pii_propagation("sales", connection_id="a")
    propagate.assert_called_once_with(connection_id="a")


def test_missing_source_is_rejected_before_pii_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = Mock(side_effect=AssertionError("source required before metadata"))
    monkeypatch.setattr(service, "get_tag_by_name", lookup)
    with pytest.raises(ValueError, match="connection_id"):
        service.propagate_pii_to_attributes(connection_id="")
    lookup.assert_not_called()
