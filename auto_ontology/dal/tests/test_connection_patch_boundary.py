# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.dal import datasources


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "name",
        "display_label",
        "connection",
        "credential_ref",
        "configuration_origin",
    ],
)
def test_generic_database_patch_cannot_change_connection_lifecycle(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    db = Mock()
    db.query_read.return_value = [{"id": "source", "name": "sales"}]
    monkeypatch.setattr(datasources, "store", lambda: db)
    with pytest.raises(ValueError, match="connection"):
        datasources.patch_catalog_node("source", {field: "replacement"})
    db.query_write.assert_not_called()
