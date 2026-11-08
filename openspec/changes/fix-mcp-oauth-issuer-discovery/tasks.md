<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Tasks

## 1. Exact MCP issuer

- [x] 1.1 Reproduce the mismatch with public GET requests and observe both opt-in runtime regressions fail: wrong advertised issuer and issuer-specific discovery HTTP 404.
- [x] 1.2 Append `/api/auth` to the selected normalized frontend base in the shared MCP auth builder; verify the new metadata/default regressions fail before the fix and pass afterward, and run the full MCP suite (101 passed).
- [x] 1.3 Document the frontend base versus issuer distinction in `docs/mcp.md` and the `.env.example` comment; verify no setting or credential value in `.env` is changed.

## 2. Issuer-specific frontend discovery

- [x] 2.1 Re-export the existing metadata GET handler at `/.well-known/oauth-authorization-server/api/auth`, retain the root alias, and verify both frontend files pass Prettier formatting and IDE error diagnostics.
- [x] 2.2 Add opt-in local runtime regressions for exact advertised issuer and RFC 8414 discovery; observe both fail against the old running services and ensure normal offline runs skip them.

## 3. Integrated deployment and verification

- [x] 3.1 Build and recreate only frontend and MCP with `--no-deps`; verify both containers are healthy without running migrations or recreating dependencies. The user explicitly authorized resuming the previously cancelled build; a transient TLS download failure cleared on one retry.
- [x] 3.2 Run `AUTO_ONTOLOGY_TEST_RUNTIME=1 .venv/bin/python -m pytest dev_tools/tests/test_oauth_runtime.py -q` successfully (2 passed); confirm root and issuer-specific discovery return matching metadata.
- [x] 3.3 Run the MCP suite (101 passed), offline Compose/runtime regressions (3 passed, 2 intentionally skipped), Ruff, Prettier, `git diff --check`, and OpenSpec 1.16.1 strict validation; verify `.env` and dependency container identities/start times remain unchanged, and report no commits.

## Workflow follow-up

- The user reconnects the OpenCode MCP connection and completes interactive login if requested; discovery checks do not claim browser sign-in was performed.
- Leave this change unarchived for review; do not commit or push.

## Verification scope

- On 2026-11-07, both live regressions failed against the old containers before rebuilding and passed after deploying the fix. The full MCP suite passed again (101 tests).
- Earlier temporary evidence was no longer present when work resumed. Fresh before/after hashes and container snapshots verify preservation during the resumed work: `.env`, HEAD, and the Git index are unchanged; only frontend/MCP containers changed, including no changes to migration job containers. Evidence is under `/tmp/opencode/auto-ontology-compose-mcp/` and is temporary.
- Ruff, Prettier, IDE error diagnostics, and OpenSpec 1.16.1 strict validation passed. `crapper`, `dryer`, and `mutator` are unavailable; those checks and interactive browser login were not performed.
