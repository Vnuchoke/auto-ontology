# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model YAML export and import.

Two things carry the weight here.

**The round trip must be idempotent.** Export a catalog, import it back, import
it again — the second import must create nothing. That is the plan's Done
criterion, and it exercises the `imported_id` matching that the
whole import turns on: an entity is found by the YAML id against `imported_id`
*or* the live id, so a document lands once however many times it is applied.

**Bug 3.** The catalog stores `'YES'`/`'NO'` and the old reader was `bool(...)`,
so every column in every export claimed to be nullable. It is asserted directly
against both stored values, because the failure was invisible — a bool is what
the schema expects, and `True` is a plausible one.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import String, or_, select  # noqa: E402

from auto_ontology.dal import model_interchange as mi  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.dal.tests.test_connection_identity_schema import (  # noqa: E402
    ROOT,
    identity_database,  # noqa: F401
)
from auto_ontology.semantic.constants import SEMANTIC_SOURCE  # noqa: E402


@pytest.fixture(autouse=True)
def require_schema(identity_database):  # noqa: F811
    from alembic import command
    from alembic.config import Config
    from auto_ontology.dal.session import dispose_engine

    dispose_engine()
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    try:
        yield
    finally:
        dispose_engine()


def _add(table, **values) -> str:
    if table is s.catalog_database:
        values.setdefault("display_label", values["name"])
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


def _link(table, **values) -> None:
    store().query_write(table.insert().values(**values))


class World:
    """A small but complete catalog: two tables, a key, a term, an attribute."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.database = _add(s.catalog_database, name=prefix)
        self.schema = _add(s.catalog_schema, database_id=self.database, name="public")

        self.orders = _add(
            s.catalog_table,
            schema_id=self.schema,
            name="orders",
            description="an order",
            table_type="base table",
            pk=["id"],
        )
        self.customers = _add(s.catalog_table, schema_id=self.schema, name="customers")
        self.order_id = _add(
            s.catalog_column,
            table_id=self.orders,
            name="id",
            data_type="integer",
            ordinal_position=1,
            is_nullable=False,
        )
        self.customer_ref = _add(
            s.catalog_column,
            table_id=self.orders,
            name="customer_id",
            data_type="integer",
            ordinal_position=2,
            is_nullable=True,
        )
        self.customer_id = _add(
            s.catalog_column,
            table_id=self.customers,
            name="id",
            data_type="integer",
            ordinal_position=1,
            is_nullable=False,
        )
        _link(
            s.column__foreign_key,
            source_column_id=self.customer_ref,
            target_column_id=self.customer_id,
        )
        _link(
            s.table__join,
            source_table_id=self.orders,
            target_table_id=self.customers,
            join_columns=[{"source": "customer_id", "target": "id"}],
        )

        self.term = _add(
            s.term,
            name=f"{prefix}-Order",
            description="a purchase",
            source=SEMANTIC_SOURCE,
        )
        _link(s.table__term, table_id=self.orders, term_id=self.term)
        self.attribute = _add(
            s.column_attribute,
            name=f"{prefix}-order-id",
            description="the identifier",
            source_column="id",
            term_name=f"{prefix}-Order",
            table_id=self.orders,
        )
        _link(
            s.column__has_attribute,
            column_id=self.order_id,
            attribute_id=self.attribute,
        )
        _link(s.column_attribute__term, attribute_id=self.attribute, term_id=self.term)
        _link(
            s.column__semantic_fk,
            column_id=self.customer_ref,
            attribute_id=self.attribute,
        )


def _cleanup(prefix: str) -> None:
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.name.like(f"{prefix}%"))
    )
    for table in (s.term, s.column_attribute, s.sql_attribute, s.custom_analysis):
        store().query_write(table.delete().where(table.c.name.like(f"{prefix}%")))


@pytest.fixture
def world():
    prefix = f"m-{uuid.uuid4().hex[:8]}"
    w = World(prefix)
    yield w
    _cleanup(prefix)


def _reid(document, seed: str):
    """Rewrite every id, as an export from another deployment would look.

    A document exported here carries this store's live ids, and the import
    matches those against `catalog_table.id` — so re-importing it *adopts* the
    rows rather than copying them, which is correct and is what
    `test_importing_onto_the_source_catalog_adopts_it` asserts. To exercise the
    create path the ids have to be foreign, which is exactly what a document
    from a different deployment has.
    """
    mapping: dict[str, str] = {}

    def new(old: str) -> str:
        return mapping.setdefault(old, f"{seed}-{uuid.uuid4().hex[:8]}")

    for database in document.data_layer.databases:
        database.id = new(database.id)
        database.display_label = seed
        for schema in database.schemas:
            schema.id = new(schema.id)
            for table in schema.tables:
                table.id = new(table.id)
                for column in table.columns:
                    column.id = new(column.id)
    for fk in document.data_layer.foreign_keys:
        fk.source_column_id = new(fk.source_column_id)
        fk.target_column_id = new(fk.target_column_id)
    for join in document.data_layer.joins:
        join.source_table_id = new(join.source_table_id)
        join.target_table_id = new(join.target_table_id)
    for term in document.semantic_layer.terms:
        term.id = new(term.id)
        term.represents = [new(table_id) for table_id in term.represents]
        # Names are re-seeded too. `term` is UNIQUE(name, source) and
        # `column_attribute` has a five-part merge key, so a document whose
        # entities merely *look* like existing ones collides — which is the
        # constraint doing its job, and is pinned separately below rather than
        # fought here. A genuinely foreign document has foreign names.
        term.name = f"{seed}-{term.name}"
        for attr in term.columns_attributes:
            attr.id = new(attr.id)
            attr.column_id = new(attr.column_id)
            attr.name = f"{seed}-{attr.name}"
    for fk in document.semantic_layer.semantic_fks:
        fk.column_id = new(fk.column_id)
        fk.column_attribute_id = new(fk.column_attribute_id)
    return document


def _document(world: World):
    rows = mi.fetch_export_rows([world.database])
    return mi.assemble_export_document(
        rows,
        dialect_by_db_name={world.database: "postgres"},
        sql_column_resolver=lambda sql, database_name: [],
    )


# --------------------------------------------------------------------------
# Bug 3 — nullability
# --------------------------------------------------------------------------


def test_nullability_survives_the_export(world) -> None:
    """Bug 3, asserted against both stored values.

    `bool('NO')` is `True`, so before this every column exported as nullable —
    and nothing failed, because a bool is exactly what the schema expects.
    """
    document = _document(world)
    columns = {
        column.name: column
        for database in document.data_layer.databases
        for schema in database.schemas
        for table in schema.tables
        if table.name == "orders"
        for column in table.columns
    }
    assert columns["id"].is_nullable is False
    assert columns["customer_id"].is_nullable is True


def test_an_undetermined_column_exports_as_nullable() -> None:
    """NULL means the connector could not tell, and absent has always meant
    nullable — the permissive reading for a constraint we cannot prove.

    The export schema types ``is_nullable`` as a plain ``bool``, so this is the
    one place the store's three states collapse into two.
    """
    assert mi._nullable_or_default(None) is True
    assert mi._nullable_or_default(True) is True
    assert mi._nullable_or_default(False) is False


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_export_carries_the_catalog(world) -> None:
    document = _document(world)
    database = document.data_layer.databases[0]

    assert database.id == world.database
    assert database.dialect == "postgres"
    tables = {table.name: table for table in database.schemas[0].tables}
    assert set(tables) == {"orders", "customers"}
    assert tables["orders"].description == "an order"
    assert tables["orders"].pk == ["id"]
    assert [c.name for c in tables["orders"].columns] == ["id", "customer_id"]


def test_export_carries_keys_joins_and_semantics(world) -> None:
    document = _document(world)

    assert [
        (fk.source_column_id, fk.target_column_id)
        for fk in document.data_layer.foreign_keys
    ] == [(world.customer_ref, world.customer_id)]
    assert [
        (j.source_table_id, j.target_table_id, j.join_columns)
        for j in document.data_layer.joins
    ] == [(world.orders, world.customers, [{"source": "customer_id", "target": "id"}])]

    term = document.semantic_layer.terms[0]
    assert term.id == world.term
    assert term.represents == [world.orders]
    assert [a.id for a in term.columns_attributes] == [world.attribute]
    assert [
        (fk.column_id, fk.column_attribute_id, fk.use_in_join_paths)
        for fk in document.semantic_layer.semantic_fks
    ] == [(world.customer_ref, world.attribute, True)]


def test_an_empty_id_list_exports_everything_not_nothing(world) -> None:
    """

    Reading it the other way would silently produce an empty document, which
    looks like a successful export of a catalog with nothing in it.
    """
    ids = {
        database.id
        for database in mi.assemble_export_document(
            mi.fetch_export_rows([]),
            dialect_by_db_name={},
            sql_column_resolver=lambda sql, database_name: [],
        ).data_layer.databases
    }
    assert world.database in ids
    # A *second* database, created here rather than assumed: the fixture builds
    # exactly one, so asserting `> 1` against whatever else happens to be in the
    # store passes only on a developer's dirty database and fails on a clean one.
    other = _add(s.catalog_database, name=f"{world.prefix}-second")
    # Down to a column: the export walks the whole hierarchy and joins through
    # catalog_column, so a database with no columns does not appear at all.
    other_schema = _add(s.catalog_schema, database_id=other, name="shop")
    other_table = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")
    _add(
        s.catalog_column,
        table_id=other_table,
        name="id",
        data_type="integer",
        ordinal_position=1,
    )
    try:
        ids_again = {
            database.id
            for database in mi.assemble_export_document(
                mi.fetch_export_rows([]),
                dialect_by_db_name={},
                sql_column_resolver=lambda sql, database_name: [],
            ).data_layer.databases
        }
        assert {world.database, other} <= ids_again
    finally:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == other)
        )


def test_validate_database_ids_rejects_an_unknown_one(world) -> None:
    mi.validate_database_ids([world.database])
    with pytest.raises(mi.UnknownDatabaseIdsError) as raised:
        mi.validate_database_ids([world.database, "no-such-database"])
    assert raised.value.database_ids == ["no-such-database"]


def test_validate_accepts_an_empty_list() -> None:
    """Empty means everything, so there is nothing to validate."""
    mi.validate_database_ids([])


def test_a_join_leaving_the_scope_is_not_exported(world) -> None:
    """Otherwise the document carries a join the importer cannot resolve."""
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")
    _link(
        s.table__join,
        source_table_id=world.orders,
        target_table_id=outside,
        join_columns=[{"source": "x", "target": "y"}],
    )

    document = _document(world)
    targets = {join.target_table_id for join in document.data_layer.joins}
    assert outside not in targets
    assert world.customers in targets


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_importing_an_export_into_an_empty_store_recreates_it(world) -> None:
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target
    document.semantic_layer.semantic_fks[0].use_in_join_paths = False

    try:
        summary = mi.apply_import_model(document, replace=False)

        assert summary["created"]["databases"] == 1
        assert summary["created"]["tables"] == 2
        assert summary["created"]["columns"] == 3
        assert summary["created"]["terms"] == 1
        assert summary["created"]["column_attributes"] == 1

        copied = store().query_read(
            select(s.catalog_database.c.id).where(s.catalog_database.c.name == target)
        )
        assert copied
        copied_semantic_fk = store().query_read(
            select(s.column__semantic_fk.c.use_in_join_paths)
            .select_from(
                s.column__semantic_fk.join(
                    s.catalog_column,
                    s.catalog_column.c.id == s.column__semantic_fk.c.column_id,
                )
            )
            .where(
                s.catalog_column.c.imported_id
                == document.semantic_layer.semantic_fks[0].column_id
            )
        )
        assert copied_semantic_fk == [{"use_in_join_paths": False}]
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_the_round_trip_is_idempotent(world) -> None:
    """Idempotence.

    A second import of the same document must create nothing. This is what
    `imported_id` buys, and it is the difference between a re-import being a
    no-op and a re-import doubling the catalog.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target

    try:
        mi.apply_import_model(document, replace=False)
        second = mi.apply_import_model(document, replace=False)

        assert sum(second["created"].values()) == 0, second["created"]
        assert second["skipped"]["databases"] == 1
        assert second["skipped"]["tables"] == 2
        assert second["skipped"]["columns"] == 3
        assert second["skipped"]["terms"] == 1
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_importing_onto_the_source_catalog_adopts_it(world) -> None:
    """The YAML ids *are* the live ids here, so nothing should be created.

    This is the `n.id = imported_id` half of the match, and the reason a first
    import onto an existing catalog does not duplicate every row.
    """
    summary = mi.apply_import_model(_document(world), replace=False)
    assert sum(summary["created"].values()) == 0, summary["created"]


def test_reimport_updates_semantic_fk_join_path_selection(world) -> None:
    document = _document(world)
    document.semantic_layer.semantic_fks[0].use_in_join_paths = False

    mi.apply_import_model(document, replace=False)

    selected = store().query_read(
        select(s.column__semantic_fk.c.use_in_join_paths).where(
            s.column__semantic_fk.c.column_id == world.customer_ref,
            s.column__semantic_fk.c.attribute_id == world.attribute,
        )
    )
    assert selected == [{"use_in_join_paths": False}]


def test_an_adopted_row_is_stamped_so_the_next_import_is_faster(world) -> None:
    mi.apply_import_model(_document(world), replace=False)
    row = store().query_read(
        select(s.catalog_table.c.imported_id).where(
            s.catalog_table.c.id == world.orders
        )
    )[0]
    assert row["imported_id"] == world.orders


def test_nullability_survives_the_whole_round_trip(world) -> None:
    """Bug 3 again, end to end: export, import, re-export.

    The export path alone can look correct while the import path writes a value
    that reads back inverted; only a round trip catches that.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target

    try:
        mi.apply_import_model(document, replace=False)
        copied_id = store().query_read(
            select(s.catalog_database.c.id).where(s.catalog_database.c.name == target)
        )[0]["id"]

        reexported = mi.assemble_export_document(
            mi.fetch_export_rows([copied_id]),
            dialect_by_db_name={},
            sql_column_resolver=lambda sql, database_name: [],
        )
        columns = {
            column.name: column
            for schema in reexported.data_layer.databases[0].schemas
            for table in schema.tables
            if table.name == "orders"
            for column in table.columns
        }
        assert columns["id"].is_nullable is False
        assert columns["customer_id"].is_nullable is True
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_an_unresolvable_reference_is_rejected(world) -> None:
    """A payload naming an id it does not carry must fail, not half-apply."""
    document = _document(world)
    document.data_layer.foreign_keys[0].target_column_id = "no-such-column"

    with pytest.raises(mi.ModelImportValidationError, match="foreign-key"):
        mi.apply_import_model(document, replace=False)


def test_a_failed_import_leaves_nothing_behind(world) -> None:
    """The whole import is one transaction — the simplification Phase 10 banked.

    A failure anywhere rolls back everything, so an import cannot leave a
    catalog with half its semantics on top.
    """
    target = f"{world.prefix}-copy"
    document = _reid(_document(world), target)
    document.data_layer.databases[0].schemas[0].database_name = target
    # A structural reference, not a `represents` entry: an unresolvable foreign
    # key endpoint is a corrupt document and still raises, whereas a term
    # representing a table outside the export is expected and now skipped.
    document.data_layer.foreign_keys[0].target_column_id = "no-such-column"

    try:
        with pytest.raises(mi.ModelImportValidationError):
            mi.apply_import_model(document, replace=False)

        assert (
            store().query_read(
                select(s.catalog_database.c.id).where(
                    s.catalog_database.c.name == target
                )
            )
            == []
        )
    finally:
        _cleanup(f"{world.prefix}-copy")


# --------------------------------------------------------------------------
# replace
# --------------------------------------------------------------------------


def test_replace_drops_a_term_the_payload_omits(world) -> None:
    document = _document(world)
    stale = _add(
        s.term,
        name=f"{world.prefix}-Stale",
        source=SEMANTIC_SOURCE,
    )
    _link(s.table__term, table_id=world.orders, term_id=stale)

    mi.apply_import_model(document, replace=True)

    assert store().query_read(select(s.term.c.id).where(s.term.c.id == stale)) == []
    assert store().query_read(select(s.term.c.id).where(s.term.c.id == world.term))


def test_a_cross_database_term_stays_importable(world) -> None:
    """A scoped export must be importable on its own.

    A Term can be represented by tables in more than one database. `_scoped_ids`
    drops the out-of-scope ones at export, so the document only references
    tables it carries. Explicit out-of-document references are rejected on import.
    """
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")

    shared = _add(s.term, name=f"{world.prefix}-Shared", source=SEMANTIC_SOURCE)
    _link(s.table__term, table_id=world.orders, term_id=shared)
    _link(s.table__term, table_id=outside, term_id=shared)

    document = _document(world)
    represents = {term.id: term.represents for term in document.semantic_layer.terms}
    assert represents[shared] == [world.orders], (
        "the export carries only the table this document also carries"
    )
    assert outside not in represents[shared], (
        "the out-of-scope table is dropped, so the document imports anywhere"
    )

    mi.apply_import_model(document, replace=True)

    linked = {
        row["table_id"]
        for row in store().query_read(
            select(s.table__term.c.table_id).where(s.table__term.c.term_id == shared)
        )
    }
    assert world.orders in linked, "the in-scope table is represented"


def test_replace_keeps_a_term_that_left_this_databases_scope(world) -> None:
    """A term representing a table outside the import is not `replace`'s to drop.

    The same all-or-nothing rule the reads and the reset apply: it belongs to
    that other database too.
    """
    other_prefix = f"{world.prefix}-other"
    other_db = _add(s.catalog_database, name=other_prefix)
    other_schema = _add(s.catalog_schema, database_id=other_db, name="public")
    outside = _add(s.catalog_table, schema_id=other_schema, name="elsewhere")

    document = _document(world)

    # Created *after* the export, so the payload does not mention it at all.
    shared = _add(s.term, name=f"{world.prefix}-Shared", source=SEMANTIC_SOURCE)
    _link(s.table__term, table_id=world.orders, term_id=shared)
    _link(s.table__term, table_id=outside, term_id=shared)

    mi.apply_import_model(document, replace=True)

    assert store().query_read(select(s.term.c.id).where(s.term.c.id == shared))


def test_replace_leaves_the_catalog_alone(world) -> None:
    """`replace` is about semantics; a table the payload omits is not deleted."""
    extra = _add(s.catalog_table, schema_id=world.schema, name="untouched")
    document = _document(world)
    document.data_layer.databases[0].schemas[0].tables = [
        table
        for table in document.data_layer.databases[0].schemas[0].tables
        if table.name != "untouched"
    ]

    mi.apply_import_model(document, replace=True)

    assert store().query_read(
        select(s.catalog_table.c.id).where(s.catalog_table.c.id == extra)
    )


def test_an_import_cannot_duplicate_a_term_name(world) -> None:
    """A constraint the graph did not have, and a behaviour change.

    `term` is `UNIQUE(name, source)`, so importing a document whose term shares
    a name with an existing one raises rather than creating a duplicate.

    Left to surface: the constraint describes a genuine ambiguity between two
    terms, and swallowing it would leave a duplicate no read can choose
    between.
    """
    document = _reid(_document(world), f"{world.prefix}-copy")
    document.data_layer.databases[0].schemas[0].database_name = f"{world.prefix}-copy"
    # Same name as the term already in the store, but a foreign id.
    document.semantic_layer.terms[0].name = f"{world.prefix}-Order"

    try:
        with pytest.raises(Exception, match="uq_term_name_source|already exists"):
            mi.apply_import_model(document, replace=False)
    finally:
        _cleanup(f"{world.prefix}-copy")


def test_import_rejects_catalog_id_from_another_source_atomically(world):
    other = World(f"{world.prefix}-other")
    document = _document(world)
    document.data_layer.databases[0].schemas[0].id = other.schema
    before = mi.fetch_export_rows([other.database])
    with pytest.raises(
        mi.ModelImportValidationError, match="(?i)(scope|parent|source)"
    ):
        mi.apply_import_model(document, replace=True)
    assert mi.fetch_export_rows([other.database]) == before


def test_export_excludes_unresolved_and_foreign_sql_owners(world):
    other = World(f"{world.prefix}-other")
    for owner in (None, other.database):
        query = _add(
            s.sql_query, connection_id=owner, sql_full_query="SELECT id FROM orders"
        )
        analysis = _add(
            s.custom_analysis, connection_id=owner, name=f"{world.prefix}-{owner}"
        )
        _link(s.sql_query__table, sql_query_id=query, table_id=world.orders)
        _link(s.custom_analysis__sql, analysis_id=analysis, sql_query_id=query)
    assert mi.fetch_export_rows([world.database])["custom_analyses"] == []


def _sql_definitions(document, world):
    from auto_ontology.server.model_interchange.schemas import (
        ModelCustomAnalysis,
        ModelSqlAttribute,
    )

    sql = "SELECT id FROM public.orders"
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id=f"{world.database}-attribute",
            connection_id=world.database,
            name="order identifier",
            sql=sql,
            term_id=world.term,
            sql_column_is=[world.order_id],
        )
    )
    document.semantic_layer.custom_analyses.append(
        ModelCustomAnalysis(
            id=f"{world.database}-analysis",
            connection_id=world.database,
            name="order identifier",
            sql=sql,
            sql_column_is=[world.order_id],
        )
    )
    return document


def _source_snapshot(world):
    ids = {
        world.database,
        world.schema,
        world.orders,
        world.customers,
        world.order_id,
        world.customer_ref,
        world.customer_id,
        world.term,
        world.attribute,
    }
    for table in (s.sql_query, s.sql_attribute, s.custom_analysis):
        ids.update(
            row["id"]
            for row in store().query_read(
                select(table.c.id).where(table.c.connection_id == world.database)
            )
        )
    snapshot = {}
    for table in s.METADATA.tables.values():
        keys = [
            column
            for column in table.c
            if isinstance(column.type, String)
            and (column.name == "id" or column.name.endswith("_id"))
        ]
        if keys:
            snapshot[table.name] = sorted(
                (
                    dict(row)
                    for row in store().query_read(
                        select(table).where(or_(*(column.in_(ids) for column in keys)))
                    )
                ),
                key=repr,
            )
    return snapshot


def test_same_physical_name_sql_and_definition_names_stay_separate(world):
    other = World(f"{world.prefix}-other")
    store().query_write(
        s.catalog_database.update()
        .where(s.catalog_database.c.id.in_([world.database, other.database]))
        .values(name="sales")
    )
    for source in (world, other):
        mi.apply_import_model(
            _sql_definitions(_document(source), source), replace=False
        )
    before = _source_snapshot(other)
    for table in (s.sql_query, s.sql_attribute, s.custom_analysis):
        rows = store().query_read(
            select(table).where(
                table.c.connection_id.in_([world.database, other.database])
            )
        )
        assert len(rows) == 2
        assert len({row["id"] for row in rows}) == 2
    exported = _document(world)
    assert (
        exported.semantic_layer.sql_attributes.manual[0].connection_id == world.database
    )
    assert exported.semantic_layer.custom_analyses[0].connection_id == world.database
    mi.apply_import_model(exported, replace=True)
    assert _source_snapshot(other) == before


def test_all_declared_sql_columns_are_checked_even_when_parser_would_ignore_them(world):
    other = World(f"{world.prefix}-other")
    document = _sql_definitions(_document(world), world)
    document.semantic_layer.custom_analyses[0].sql_column_is.append(other.order_id)
    before = mi.fetch_export_rows([world.database])
    with pytest.raises(mi.ModelImportValidationError, match="foreign-source"):
        mi.apply_import_model(document, replace=True)
    assert mi.fetch_export_rows([world.database]) == before
    assert store().query_read(select(s.sql_query)) == []


def test_foreign_definition_id_cannot_be_adopted(world):
    other = World(f"{world.prefix}-other")
    foreign = _add(s.custom_analysis, connection_id=other.database, name="foreign")
    document = _sql_definitions(_document(world), world)
    document.semantic_layer.sql_attributes.manual = []
    document.semantic_layer.custom_analyses[0].id = foreign
    with pytest.raises(mi.ModelImportValidationError, match="scope"):
        mi.apply_import_model(document, replace=False)
    assert (
        store().query_read(
            select(s.custom_analysis).where(s.custom_analysis.c.id == foreign)
        )[0]["connection_id"]
        == other.database
    )


def test_export_rejects_sql_with_any_foreign_catalog_reference(world):
    other = World(f"{world.prefix}-other")
    query = _add(
        s.sql_query,
        connection_id=world.database,
        sql_full_query="SELECT id FROM orders",
    )
    analysis = _add(s.custom_analysis, connection_id=world.database, name="mixed")
    _link(s.custom_analysis__sql, analysis_id=analysis, sql_query_id=query)
    for table_id in (world.orders, other.orders):
        _link(s.sql_query__table, sql_query_id=query, table_id=table_id)
    assert mi.fetch_export_rows([world.database])["custom_analyses"] == []


def test_export_projects_shared_term_before_assembly(world):
    other = World(f"{world.prefix}-other")
    _link(s.table__term, table_id=other.orders, term_id=world.term)
    store().query_write(
        s.column_attribute__term.update()
        .where(s.column_attribute__term.c.attribute_id == other.attribute)
        .values(term_id=world.term)
    )
    terms = {
        term["id"]: term for term in mi.fetch_export_rows([world.database])["terms"]
    }
    assert terms[world.term]["represents"] == [world.orders]
    assert [attr["column_id"] for attr in terms[world.term]["columns_attributes"]] == [
        world.order_id
    ]


def test_duplicate_catalog_id_under_different_document_sources_is_rejected(world):
    document = _reid(_document(world), f"{world.prefix}-copy")
    duplicate = document.data_layer.databases[0].model_copy(deep=True)
    duplicate.id = str(uuid.uuid4())
    duplicate.display_label += " second"
    document.data_layer.databases.append(duplicate)
    with pytest.raises(mi.ModelImportValidationError, match="scope"):
        mi.apply_import_model(document, replace=False)


def test_export_checks_all_queries_of_an_owner_not_only_matching_query(world):
    other = World(f"{world.prefix}-other")
    analysis = _add(
        s.custom_analysis, connection_id=world.database, name="mixed definitions"
    )
    for source in (world, other):
        query = _add(
            s.sql_query,
            connection_id=source.database,
            sql_full_query="SELECT id FROM orders",
        )
        _link(s.custom_analysis__sql, analysis_id=analysis, sql_query_id=query)
        _link(s.sql_query__table, sql_query_id=query, table_id=source.orders)
    assert mi.fetch_export_rows([world.database])["custom_analyses"] == []


def test_new_catalog_root_keeps_explicit_id_and_requires_label(world):
    document = _reid(_document(world), f"{world.prefix}-copy")
    target = document.data_layer.databases[0]
    label = target.display_label
    target.display_label = ""
    with pytest.raises(mi.ModelImportValidationError, match="display_label"):
        mi.apply_import_model(document, replace=False)
    target.display_label = label
    result = mi.apply_import_model(document, replace=False)
    assert result["database_ids"] == [target.id]
    row = store().query_read(
        select(s.catalog_database).where(s.catalog_database.c.id == target.id)
    )[0]
    assert row["display_label"] == label
    assert row["configuration_origin"] is None


def test_import_never_uses_schema_name_or_id_as_physical_database_name(world):
    document = _reid(_document(world), f"{world.prefix}-copy")
    for schema in document.data_layer.databases[0].schemas:
        schema.database_name = ""
    with pytest.raises(mi.ModelImportValidationError, match="physical database_name"):
        mi.apply_import_model(document, replace=False)


def test_existing_target_uuid_and_curated_metadata_are_preserved(world):
    document = _document(world)
    document.data_layer.databases[0].display_label = "not an authorized rename"
    for table in document.data_layer.databases[0].schemas[0].tables:
        table.description = "not curated"
    summary = mi.apply_import_model(document, replace=False)
    assert summary["database_ids"] == [world.database]
    assert store().query_read(
        select(s.catalog_database.c.display_label).where(
            s.catalog_database.c.id == world.database
        )
    ) == [{"display_label": world.prefix}]
    assert store().query_read(
        select(s.catalog_table.c.description).where(
            s.catalog_table.c.id == world.orders
        )
    ) == [{"description": "an order"}]


def test_multisource_legacy_sql_requires_explicit_document_owner(world):
    other = World(f"{world.prefix}-other")
    document = _sql_definitions(_document(world), world)
    document.data_layer.databases.extend(_document(other).data_layer.databases)
    document.semantic_layer.custom_analyses[0].connection_id = ""
    with pytest.raises(
        mi.ModelImportValidationError, match="explicit document connection_id"
    ):
        mi.apply_import_model(document, replace=False)


@pytest.mark.parametrize("reference", ["term", "semantic_fk"])
def test_explicit_out_of_document_catalog_links_are_rejected(world, reference):
    other = World(f"{world.prefix}-other")
    document = _document(world)
    if reference == "term":
        document.semantic_layer.terms[0].represents.append(other.orders)
    else:
        document.semantic_layer.semantic_fks[0].column_id = other.order_id
    before = _source_snapshot(other)
    with pytest.raises(mi.ModelImportValidationError, match="Cannot resolve"):
        mi.apply_import_model(document, replace=False)
    assert _source_snapshot(other) == before


def test_replace_exported_import_preserves_live_ids_and_curation(world):
    from auto_ontology.server.model_interchange.schemas import (
        ModelCustomAnalysis,
        ModelSqlAttribute,
    )

    document = _reid(_document(world), f"{world.prefix}-copy")
    database = document.data_layer.databases[0]
    term = document.semantic_layer.terms[0]
    orders = next(
        table for table in database.schemas[0].tables if table.name == "orders"
    )
    column = next(column for column in orders.columns if column.name == "id")
    document.semantic_layer.sql_attributes.manual.append(
        ModelSqlAttribute(
            id="payload-sql-attribute",
            connection_id=database.id,
            name="imported identifier",
            term_id=term.id,
            sql="SELECT id FROM public.orders",
            sql_column_is=[column.id],
        )
    )
    document.semantic_layer.custom_analyses.append(
        ModelCustomAnalysis(
            id="payload-analysis",
            connection_id=database.id,
            name="imported analysis",
            sql="SELECT id FROM public.orders",
            sql_column_is=[column.id],
        )
    )
    mi.apply_import_model(document, replace=False)
    identities = {}
    tag = _add(s.tag, name="Curated import")
    for table, payload_id, tag_column, curated in (
        (
            s.term,
            term.id,
            "term_id",
            {
                "name_certified": True,
                "description_certified": True,
                "owner_id": "curator",
            },
        ),
        (
            s.column_attribute,
            term.columns_attributes[0].id,
            "column_attribute_id",
            {"certified": True},
        ),
        (
            s.sql_attribute,
            "payload-sql-attribute",
            "sql_attribute_id",
            {"certified": True, "owner_id": "curator"},
        ),
        (s.custom_analysis, "payload-analysis", None, {}),
    ):
        row = store().query_read(
            select(table).where(table.c.imported_id == payload_id)
        )[0]
        assert row["id"] != payload_id
        identities[table.name] = row["id"]
        if curated:
            store().query_write(
                table.update().where(table.c.id == row["id"]).values(**curated)
            )
        if tag_column:
            _add(s.tag_target, tag_id=tag, **{tag_column: row["id"]})
    before = {
        name: store().query_read(
            select(s.METADATA.tables[name]).where(
                s.METADATA.tables[name].c.id == live_id
            )
        )
        for name, live_id in identities.items()
    }
    tags_before = store().query_read(
        select(s.tag_target).where(s.tag_target.c.tag_id == tag)
    )
    exported = mi.assemble_export_document(
        mi.fetch_export_rows([database.id]),
        dialect_by_db_name={},
        sql_column_resolver=mi.make_cached_sql_column_resolver(),
    )
    result = mi.apply_import_model(exported, replace=True)
    assert result["created"]["sql_attributes"] == 0
    for name, live_id in identities.items():
        table = s.METADATA.tables[name]
        assert (
            store().query_read(select(table).where(table.c.id == live_id))
            == before[name]
        )
    assert (
        store().query_read(select(s.tag_target).where(s.tag_target.c.tag_id == tag))
        == tags_before
    )


def test_replace_a_retains_term_used_by_b_sql_attribute_and_all_b_reads(world):
    from auto_ontology.dal.sql_attributes import (
        list_sql_attributes,
        get_full_sql_attribute_by_id,
    )

    other = World(f"{world.prefix}-other")
    document_b = _sql_definitions(_document(other), other)
    # A global term may be represented only by A but own a SQL projection in B.
    definition = document_b.semantic_layer.sql_attributes.manual[0]
    definition.term_id = world.term
    document_b.semantic_layer.terms.append(
        _document(world)
        .semantic_layer.terms[0]
        .model_copy(deep=True, update={"represents": [], "columns_attributes": []})
    )
    mi.apply_import_model(document_b, replace=False)
    attribute_id = store().query_read(
        select(s.sql_attribute.c.id).where(
            s.sql_attribute.c.connection_id == other.database
        )
    )[0]["id"]
    term_before = store().query_read(select(s.term).where(s.term.c.id == world.term))
    b_before = _source_snapshot(other)
    list_before = list_sql_attributes(connection_id=other.database)
    full_before = get_full_sql_attribute_by_id(attribute_id)
    assert [row["id"] for row in list_before] == [attribute_id]
    assert full_before is not None
    document_a = _document(world)
    document_a.semantic_layer.terms = []
    document_a.semantic_layer.semantic_fks = []
    document_a.semantic_layer.sql_attributes.manual = []
    mi.apply_import_model(document_a, replace=True)
    assert (
        store().query_read(select(s.term).where(s.term.c.id == world.term))
        == term_before
    )
    assert _source_snapshot(other) == b_before
    assert list_sql_attributes(connection_id=other.database) == list_before
    assert get_full_sql_attribute_by_id(attribute_id) == full_before


@pytest.mark.parametrize("relation", ["column_attribute", "tag"])
def test_replace_retains_term_with_remaining_global_relations(world, relation):
    if relation == "column_attribute":
        other = World(f"{world.prefix}-other")
        store().query_write(
            s.column_attribute__term.update()
            .where(s.column_attribute__term.c.attribute_id == other.attribute)
            .values(term_id=world.term)
        )
        table = s.column_attribute__term
    else:
        tag = _add(s.tag, name="Global vocabulary curation")
        _add(s.tag_target, tag_id=tag, term_id=world.term)
        table = s.tag_target
    before = store().query_read(select(table).where(table.c.term_id == world.term))
    # The A-owned column attribute is intentionally removed; only external links survive.
    if relation == "column_attribute":
        before = [row for row in before if row["attribute_id"] == other.attribute]
    document = _document(world)
    document.semantic_layer.terms = []
    document.semantic_layer.semantic_fks = []
    mi.apply_import_model(document, replace=True)
    assert store().query_read(select(s.term).where(s.term.c.id == world.term))
    assert (
        store().query_read(select(table).where(table.c.term_id == world.term)) == before
    )


@pytest.mark.parametrize("identity", ["live", "imported"])
def test_replace_keeps_imported_term_without_other_retaining_relations(world, identity):
    document = _reid(_document(world), f"{world.prefix}-copy")
    term = document.semantic_layer.terms[0]
    term.columns_attributes = []
    document.semantic_layer.semantic_fks = []
    mi.apply_import_model(document, replace=False)
    row = store().query_read(select(s.term).where(s.term.c.imported_id == term.id))[0]
    assert row["id"] != term.id
    store().query_write(
        s.term.update()
        .where(s.term.c.id == row["id"])
        .values(name_certified=True, owner_id="curator")
    )
    before = store().query_read(select(s.term).where(s.term.c.id == row["id"]))
    if identity == "live":
        document = mi.assemble_export_document(
            mi.fetch_export_rows([document.data_layer.databases[0].id]),
            dialect_by_db_name={},
            sql_column_resolver=lambda *_args: [],
        )
    mi.apply_import_model(document, replace=True)
    assert store().query_read(select(s.term).where(s.term.c.id == row["id"])) == before
