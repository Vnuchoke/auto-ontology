# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The integration head retains both migration branches and schema compatibility."""

from pathlib import Path
from unittest.mock import Mock

from alembic.config import Config
from alembic.script import ScriptDirectory
import yaml


ROOT = Path(__file__).resolve().parents[2]
BRANCH_HEADS = {"caee572fa81c", "e38c06a17f42"}
MERGE_REVISION = "f4a39d8b2160"
COMPATIBILITY_REVISION = "f5b40e9c3271"
CONCEPT_REVISION = "38a47b1ef206"
MASTER_CONCEPT_REVISION = "0b6f3c2a9d14"
CONCEPT_MERGE = "715d9c03a842"
CONVERSATION_HEAD = "e7a3d5c19f42"
HEAD = "2d718c2f16a4"


def test_merged_head_preserves_both_migration_branches_and_helm_pin() -> None:
    scripts = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    heads = scripts.get_heads()
    assert len(heads) == 1, f"expected one merged head, found {heads}"
    head = scripts.get_revision(heads[0])
    assert head.revision == HEAD
    assert set(head.down_revision) == {CONCEPT_MERGE, CONVERSATION_HEAD}
    assert set(scripts.get_revision(CONCEPT_MERGE).down_revision) == {
        CONCEPT_REVISION,
        MASTER_CONCEPT_REVISION,
    }
    assert scripts.get_revision(MASTER_CONCEPT_REVISION).down_revision == "caee572fa81c"
    assert (
        scripts.get_revision(CONCEPT_REVISION).down_revision == COMPATIBILITY_REVISION
    )
    assert scripts.get_revision(COMPATIBILITY_REVISION).down_revision == MERGE_REVISION
    assert set(scripts.get_revision(MERGE_REVISION).down_revision) == BRANCH_HEADS
    ancestors = {revision.revision for revision in scripts.walk_revisions()}
    assert BRANCH_HEADS <= ancestors
    values = yaml.safe_load((ROOT / "helm/auto-ontology/values.yaml").read_text())
    assert values["backend"]["alembicRevision"] == head.revision


def test_merge_revision_does_not_execute_database_operations(monkeypatch) -> None:
    scripts = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    heads = scripts.get_heads()
    assert len(heads) == 1, f"expected one merged head, found {heads}"
    revision = scripts.get_revision(MERGE_REVISION)
    assert set(revision.down_revision) == BRANCH_HEADS
    operations = Mock()
    monkeypatch.setattr("alembic.op", operations)
    revision.module.upgrade()
    revision.module.downgrade()
    assert operations.mock_calls == []
