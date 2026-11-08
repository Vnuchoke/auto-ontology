<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design

## Context

See [proposal.md](proposal.md) for motivation and [the discovery contract](specs/mcp-oauth-discovery/spec.md) for required behavior. Live responses confirmed an origin-level advertised authorization server, Better Auth metadata with `/api/auth` as issuer, and a missing issuer-specific RFC 8414 route. The previous tests checked origin separation but incorrectly treated the origin as the issuer.

The user requested this change after implementation had begun. This document records the selected approach and completed evidence honestly; it does not imply the artifacts preceded those edits.

## Goals / Non-Goals

- Keep `AUTO_ONTOLOGY_AUTH_URL` as a frontend base URL; callers must not add `/api/auth` themselves.
- Correct defaults and public overrides in the shared MCP auth builder, without changing HTTP clients or token verification.
- Reuse the existing frontend metadata handler rather than constructing another metadata document.
- Do not change Better Auth's issuer, database contents, ports, credentials, source connections, or OpenCode's issuer validation.

## Decisions

1. **Append `/api/auth` in the MCP auth builder.** Normalize trailing slashes on the selected base first. This covers both explicit public URLs and the default API-base fallback without another environment variable. Merely changing Compose would leave standalone/default callers broken.
2. **Add an explicit RFC 8414 alias.** `frontend/app/.well-known/oauth-authorization-server/api/auth/route.ts` re-exports the existing `GET` handler. Relying on a particular client's OIDC fallback would leave the normative OAuth discovery URL returning 404. Rewriting the returned issuer to the origin could conflict with other authorization responses and is rejected.
3. **Keep two regression layers.** ASGI tests pin exact MCP metadata and existing mocked tests pin private UserInfo/tool routing. Opt-in local runtime tests cover the real frontend discovery document and its relationship to MCP metadata without performing login, registration, or data queries.

## Risks / Trade-offs

- Rebuilding services causes brief frontend/MCP downtime → rebuild first, then recreate only those services with `--no-deps`; do not run migrations or recreate dependencies.
- The new frontend alias is not present in older images → deploy frontend and MCP together, not just MCP.
- Cached client metadata can retain the old origin → reconnect the MCP server after the runtime metadata checks succeed; do not disable issuer validation.
- Runtime tests assume the agreed local Compose ports → keep them opt-in with `AUTO_ONTOLOGY_TEST_RUNTIME=1`, not mandatory network calls during ordinary unit tests.
- Successful discovery is not proof of a complete browser login → report the boundary explicitly; no automated login or token persistence is authorized here.
- A cancelled build requires renewed authorization → the user explicitly authorized resuming frontend/MCP deployment and live verification; do not infer permission from a cancellation notice.

## Migration Plan

1. Build only `auto-ontology-frontend` and `auto-ontology-mcp` using existing Dockerfiles.
2. Recreate only those services with `docker compose up -d --no-deps --wait auto-ontology-frontend auto-ontology-mcp`.
3. Run the opt-in runtime regressions; compare the origin-level alias and issuer-specific document; confirm exact issuer equality with MCP metadata.
4. Verify `.env` is unchanged and the Postgres, backend, and ingestion container identities/start times are unchanged.
5. If deployment fails, retain the error evidence and restore only these two services to their previous local images; do not remove volumes, reset the worktree, or run migrations.

No commit, push, or archive is included.

OpenSpec artifacts are validated with the user-selected OpenSpec 1.16.1.
