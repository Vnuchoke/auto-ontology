<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Proposal

## Why

OpenCode rejects the running MCP server because it advertises the frontend origin as its authorization server, while Better Auth's metadata identifies the issuer as that origin plus `/api/auth`. The RFC 8414 discovery URL for that issuer also returns 404, so changing only the advertised URL is not a complete standards-based fix.

## What Changes

- Advertise Better Auth's exact `/api/auth` issuer in MCP protected-resource metadata, using the public frontend base URL when configured.
- Serve the existing authorization-server metadata at the issuer-specific RFC 8414 discovery path, retaining the origin-level alias.
- Preserve internal API and UserInfo routing, the MCP resource URL, authentication requirements, existing environment variables, and loopback port mappings.
- Add issuer/discovery regressions and document the public base URL versus issuer distinction.

## Capabilities

### New Capabilities

- `mcp-oauth-discovery`: Consistent public issuer and standards-based metadata discovery for MCP clients, independent of private API routing. This establishes the first specification of existing behavior; it does not introduce a new identity provider.

### Modified Capabilities

None; the main specification inventory is currently empty.

## Impact

- MCP authorization-server metadata and its Python regression tests.
- A frontend discovery-route alias reusing the existing Better Auth handler.
- Local Compose runtime regression tests and MCP documentation.
- No new runtime dependencies, migrations, source-database operations, reranker work, credential changes, or OpenCode configuration changes.
- Rollout may rebuild and recreate only frontend and MCP, as authorized. No commits, pushes, or archival are part of this change.
