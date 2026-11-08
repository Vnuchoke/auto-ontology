# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def same_name_sources() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[1]
        / "dev_tools/fixtures/same_name_sources.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
