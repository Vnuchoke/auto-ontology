# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.connectors.sources import SourceSelectionError
from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store
from auto_ontology.dal.tests.test_term_projections import (
    identity_database as identity_database,
    projections as projections,
)
from auto_ontology.retrieval.source_context import resolve_query_source
from auto_ontology.vdb.tests.test_source_readiness import (
    vector_catalog as vector_catalog,
)


@pytest.fixture
def active_sources(
    projections: None, vector_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONNECTION_STRINGS", raising=False)
    store().query_write(
        s.catalog_database.update().values(configuration_origin="managed")
    )


@pytest.mark.usefixtures("active_sources")
def test_one_source_cannot_borrow_another_sources_vectors() -> None:
    source = resolve_query_source(connection_id="a")
    assert (source.connection_id, source.database_name) == ("a", "sales")
    with pytest.raises(SourceSelectionError) as error:
        resolve_query_source(connection_id="b")
    assert (error.value.status_code, error.value.code) == (503, "source_not_ready")


@pytest.mark.usefixtures("active_sources")
def test_store_resolver_requires_choice_between_same_named_sources() -> None:
    with pytest.raises(SourceSelectionError) as error:
        resolve_query_source()
    assert (error.value.status_code, error.value.code) == (422, "source_required")
    with pytest.raises(SourceSelectionError) as legacy:
        resolve_query_source(target_db="sales")
    assert (legacy.value.status_code, legacy.value.code) == (409, "ambiguous_source")


@pytest.mark.usefixtures("active_sources")
def test_vectors_do_not_bless_missing_semantic_context() -> None:
    store().query_write(s.table__term.delete())
    store().query_write(s.column_attribute__term.delete())
    with pytest.raises(SourceSelectionError) as error:
        resolve_query_source(connection_label="a")
    assert (error.value.status_code, error.value.code) == (503, "source_not_ready")
