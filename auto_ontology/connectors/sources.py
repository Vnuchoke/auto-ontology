# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-free source identities; registration belongs only to startup."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import SplitResult, parse_qsl, unquote, urlsplit
from uuid import UUID, uuid4

from psycopg import Error as PsycopgError
from psycopg.conninfo import conninfo_to_dict
from sqlalchemy.exc import IntegrityError

from auto_ontology.dal import connections
from auto_ontology.dal.session import write_transaction


@dataclass(frozen=True)
class SourceDescriptor:
    connection_id: str
    display_label: str
    database_name: str
    dialect: str | None
    configuration_origin: Literal["managed", "env"]


class SourceSelectionError(ValueError):
    """Safe boundary failure: the message contains only the public error code."""

    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def resolve_source(
    *,
    connection_id: str | None = None,
    connection_label: str | None = None,
    target_db: str | None = None,
) -> SourceDescriptor:
    """Resolve exactly one active identity using metadata, never source drivers."""
    selectors = [
        value
        for value in (connection_id, connection_label, target_db)
        if value is not None
    ]
    if len(selectors) > 1 or any(
        not isinstance(value, str) or not value.strip() for value in selectors
    ):
        raise SourceSelectionError(422, "invalid_source_selector")
    try:
        active = list_active_sources()
        if not selectors:
            if not active:
                raise SourceSelectionError(503, "source_unavailable")
            if len(active) > 1:
                raise SourceSelectionError(422, "source_required")
            return active[0]
        if target_db is not None:
            target_db = target_db.strip()
            try:
                connection_id = str(UUID(target_db))
            except ValueError:
                pass
        if connection_id is not None:
            matches = [item for item in active if item.connection_id == connection_id]
        elif connection_label is not None:
            normalized = connections.normalize_source_labels(
                [connection_label, *[item.display_label for item in active]]
            )
            matches = [
                item
                for item, label in zip(active, normalized[1:])
                if label == normalized[0]
            ]
        else:
            matches = [
                item
                for item in active
                if item.database_name.casefold() == target_db.casefold()
            ]
        if not matches:
            raise SourceSelectionError(404, "source_unavailable")
        if len(matches) > 1:
            raise SourceSelectionError(409, "ambiguous_source")
        return matches[0]
    except SourceSelectionError:
        raise
    except Exception:
        raise SourceSelectionError(503, "source_not_ready") from None


@dataclass(frozen=True)
class EnvSource:
    connection_id: str | None
    display_label: str | None
    database_name: str
    dialect: str
    _driver_url: str = field(repr=False)


def _metadata_options(query: str) -> tuple[dict[str, str], str]:
    selectors: dict[str, str] = {}
    options: list[str] = []
    for part in query.split("&"):
        pairs = parse_qsl(part, keep_blank_values=True)
        if pairs and pairs[0][0] in ("metadata_connection_id", "metadata_label"):
            key, value = pairs[0]
            if key in selectors or not value.strip():
                raise ValueError
            selectors[key] = value
        else:
            # Preserve driver-specific options and percent encoding byte-for-byte.
            options.append(part)
    return selectors, "&".join(options) if any(options) else ""


def _physical_name(parsed: SplitResult, driver_url: str) -> str:
    scheme = parsed.scheme.split("+", 1)[0]
    name = unquote(parsed.path.lstrip("/"))
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if scheme == "snowflake":
        return query.get("database") or name
    if scheme in ("trino", "kyuubi"):
        return name or query.get("catalog", "")
    if scheme in ("postgres", "postgresql") and "+" not in parsed.scheme:
        return conninfo_to_dict(driver_url).get("dbname", "")
    if scheme == "sqlite":
        return Path(unquote(parsed.path)).stem
    if scheme == "duckdb":
        path = parsed.netloc + parsed.path
        return "memory" if path in ("", ":memory:") else Path(path).stem
    return name


def _parse_env_source(value: str) -> EnvSource:
    # Local import: registry consumes descriptors, not the other way around.
    from auto_ontology.connectors.registry import CONNECTOR_REGISTRY

    parsed = urlsplit(value)
    scheme = parsed.scheme.split("+", 1)[0]
    driver = CONNECTOR_REGISTRY.get(scheme)
    if driver is None or parsed.fragment:
        raise ValueError
    selectors, query = _metadata_options(parsed.query)
    driver_url = value.split("?", 1)[0] + ("?" + query if query else "")
    source_id = selectors.get("metadata_connection_id")
    if source_id is not None:
        source_id = str(UUID(source_id))
    label = selectors.get("metadata_label")
    if label is not None:
        label = label.strip()
    name = _physical_name(parsed, driver_url)
    if not name or (scheme not in ("sqlite", "duckdb") and not parsed.hostname):
        raise ValueError
    # Accessing port validates malformed network selectors without connecting.
    if scheme not in ("sqlite", "duckdb"):
        parsed.port
    # Same class-only contract as test_registry_dialects: no driver instance.
    dialect = driver.dialect.fget(None)
    return EnvSource(source_id, label, name, dialect, driver_url)


def parse_env_sources(raw: str) -> list[EnvSource]:
    """Validate selectors without opening drivers; keep other URL options intact."""
    from auto_ontology.connectors.registry import parse_connection_strings

    try:
        result = [_parse_env_source(value) for value in parse_connection_strings(raw)]
        ids = [item.connection_id for item in result if item.connection_id]
        if len(ids) != len(set(ids)):
            raise ValueError
        counts = Counter(item.database_name for item in result)
        if any(
            counts[item.database_name] > 1
            and (item.connection_id is None or item.display_label is None)
            for item in result
        ):
            raise ValueError
        if raw.strip() and not result:
            raise ValueError
    except (ValueError, TypeError, PsycopgError):
        raise ValueError(
            "invalid_source_selector: invalid environment source configuration"
        ) from None
    return result


def _resolve_env(
    desired: list[EnvSource], *, registering: bool
) -> list[SourceDescriptor]:
    rows = connections.list_source_metadata()
    by_id = {row["id"]: row for row in rows}
    resolved: list[SourceDescriptor] = []
    for item in desired:
        existing = by_id.get(item.connection_id) if item.connection_id else None
        if item.connection_id is None:
            matches = [row for row in rows if row["name"] == item.database_name]
            if len(matches) > 1:
                raise ValueError("source_conflict: explicit connection ID required")
            existing = matches[0] if matches else None
        if existing is not None:
            if (
                existing["configuration_origin"] != "env"
                or existing["name"] != item.database_name
            ):
                raise ValueError(
                    "source_not_ready: environment source ownership or physical name conflicts; explicit migration mapping required"
                )
            source_id = existing["id"]
            label = (
                item.display_label
                if item.display_label is not None
                else existing["display_label"]
            )
            if not registering and label != existing["display_label"]:
                raise ValueError(
                    "source_not_ready: environment metadata requires registration"
                )
        else:
            if not registering:
                raise ValueError(
                    "source_not_ready: environment source is not registered"
                )
            source_id = item.connection_id or str(uuid4())
            label = item.display_label or item.database_name
        resolved.append(
            SourceDescriptor(source_id, label, item.database_name, item.dialect, "env")
        )
    # PostgreSQL defines label equivalence (not Python's Unicode casefold).
    normalized = connections.normalize_source_labels(
        [item.display_label for item in resolved]
    )
    if len(normalized) != len(set(normalized)) or len(
        {item.connection_id for item in resolved}
    ) != len(resolved):
        raise ValueError("source_conflict: duplicate source ID or display label")
    desired_ids = {item.connection_id for item in resolved}
    if any(
        row["id"] not in desired_ids and row["normalized_label"] in normalized
        for row in rows
    ):
        raise ValueError("source_conflict: display label is unavailable")
    return resolved


def bootstrap_sources() -> list[SourceDescriptor]:
    """Atomically register the entire env batch, never its credentials."""
    raw = os.environ.get("CONNECTION_STRINGS", "")
    if not raw.strip():
        return []
    desired = parse_env_sources(raw)
    try:
        with write_transaction():
            connections.lock_source_registration()
            resolved = _resolve_env(desired, registering=True)
            for item in resolved:
                connections.register_env_source(
                    connection_id=item.connection_id,
                    display_label=item.display_label,
                    database_name=item.database_name,
                )
        return resolved
    except IntegrityError:
        raise ValueError(
            "source_conflict: source registration conflicts with existing metadata"
        ) from None


def list_active_sources() -> list[SourceDescriptor]:
    """Read metadata only; missing env registration fails closed, without writes."""
    raw = os.environ.get("CONNECTION_STRINGS", "")
    if raw.strip():
        return _resolve_env(parse_env_sources(raw), registering=False)
    return [
        SourceDescriptor(
            row["id"], row["display_label"], row["name"], row["dialect"], "managed"
        )
        for row in connections.list_source_metadata()
        if row["configuration_origin"] == "managed"
    ]
