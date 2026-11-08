# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from auto_ontology.connectors.sources import SourceDescriptor
from auto_ontology.semantic import (
    compile as compilation,
    pipeline,
    semantic_fk,
    bridge_tables,
    sql_attribute_suggester,
    domain,
)


def source(cid: str) -> SourceDescriptor:
    return SourceDescriptor(cid, cid, "sales", "postgres", "managed")


@pytest.mark.parametrize(
    "entry",
    [
        compilation.run_semantic_compilation,
        pipeline.compile_semantic_layer,
        semantic_fk.resolve_semantic_fks,
        bridge_tables.build_bridge_tables_sql_attributes,
        sql_attribute_suggester.suggest_sql_attributes,
    ],
)
@pytest.mark.parametrize("cid", ["unknown", "inactive"])
def test_invalid_source_fails_before_models_or_sampling(monkeypatch, entry, cid):
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        lambda: [source("a"), source("b")],
    )
    model = Mock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(compilation, "build_semantic_embedder", model)
    monkeypatch.setattr(pipeline, "reset_sampling_breaker", model)
    monkeypatch.setattr(semantic_fk, "get_connector", model)
    monkeypatch.setattr(bridge_tables, "fetch_bridge_table_candidates", model)
    monkeypatch.setattr(sql_attribute_suggester, "fetch_terms_with_sqls", model)
    with pytest.raises(ValueError, match="active"):
        entry(cid)
    model.assert_not_called()


def test_pipeline_propagates_selected_id_and_physical_name(monkeypatch):
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        lambda: [source("a"), source("b")],
    )
    tables = Mock(return_value=[dict(id="t-a", name="orders", connection_id="a")])
    ctx = dict(columns=[dict(name="marker_a")], connection_id="a")
    fetch_ctx = Mock(return_value=ctx)
    process = Mock(return_value=None)
    breaker = Mock()
    monkeypatch.setattr(pipeline, "fetch_all_tables_without_term", tables)
    monkeypatch.setattr(pipeline, "fetch_table_context", fetch_ctx)
    monkeypatch.setattr(pipeline, "process_table", process)
    monkeypatch.setattr(pipeline, "reset_sampling_breaker", breaker)
    monkeypatch.setattr(pipeline, "load_domain_summary", lambda cid: None)
    monkeypatch.setattr(pipeline, "is_distinct_value_probing_enabled", lambda: False)
    pipeline.compile_semantic_layer("a")
    tables.assert_called_once_with(connection_id="a")
    fetch_ctx.assert_called_once_with("t-a", connection_id="a")
    breaker.assert_called_once_with("a")
    assert process.call_args.kwargs["connection_id"] == "a"
    assert process.call_args.kwargs["database_name"] == "sales"


def test_all_compilation_stages_receive_identity_not_physical_name(monkeypatch):
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        lambda: [source("a"), source("b")],
    )
    embed = Mock(return_value=None)
    monkeypatch.setattr(compilation, "build_semantic_embedder", embed)
    monkeypatch.setattr(compilation, "load_domain_summary", lambda cid: None)
    stages = []
    for module, name in [
        (compilation, "compile_semantic_layer"),
        (compilation, "resolve_semantic_fks"),
        (compilation, "build_bridge_tables_sql_attributes"),
        (sql_attribute_suggester, "suggest_sql_attributes"),
    ]:
        stage = Mock(return_value=2)
        monkeypatch.setattr(module, name, stage)
        stages.append(stage)
    assert compilation.run_semantic_compilation("a") == 2
    embed.assert_called_once_with("sales", connection_id="a", reset=False)
    assert [stage.call_args.args for stage in stages] == [("a",)] * 4


def test_summary_cache_uses_path_safe_identity_without_name_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(domain, "_SUMMARY_DIR", tmp_path)
    (tmp_path / "sales.json").write_text('{"domains":["wrong-source"]}')
    (tmp_path / "a%2Fb.json").write_text('{"domains":["source-a"]}')
    assert domain.load_domain_summary("a/b").domains == ["source-a"]
    assert domain.load_domain_summary("b") is None


def test_fk_connector_selection_is_id_only(monkeypatch):
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.list_active_sources",
        lambda: [source("a"), source("b")],
    )
    connector = Mock()
    factory = Mock(return_value=connector)
    monkeypatch.setattr(semantic_fk, "get_connector", factory)
    assert semantic_fk._resolve_connector("a") is connector
    factory.assert_called_once_with("a")


def test_fk_model_never_sees_foreign_source_hit(monkeypatch):
    retriever = Mock()
    retriever.query.return_value = [
        dict(metadata=dict(id="foreign-attr", connection_id="b", table_id="t-b"))
    ]
    judge = Mock(return_value="foreign-attr")
    monkeypatch.setattr(semantic_fk, "_llm_pick_hit", judge)
    assert (
        semantic_fk._resolve_via_vdb(dict(id="c-a", table_id="t-a"), retriever, "a")
        is None
    )
    judge.assert_not_called()


def test_cli_bootstraps_then_compiles_by_id(monkeypatch):
    from auto_ontology.semantic.__main__ import main

    calls = []
    monkeypatch.setattr("sys.argv", ["semantic", "--connection-id", "a"])
    monkeypatch.setattr(
        "auto_ontology.dal.schema_version.require_current_schema",
        lambda: calls.append("schema"),
    )
    monkeypatch.setattr(
        "auto_ontology.connectors.sources.bootstrap_sources",
        lambda: calls.append("bootstrap"),
    )
    monkeypatch.setattr(
        compilation, "run_semantic_compilation", lambda cid: calls.append(cid)
    )
    main()
    assert calls == ["schema", "bootstrap", "a"]
