<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spec Delta

## Purpose

Allow MCP clients to discover the application's authorization server with an exact issuer match while keeping private service-to-service API traffic separate from browser-facing URLs.

## ADDED Requirements

### Requirement: Advertise the application's exact OAuth issuer

MCP protected-resource metadata SHALL advertise an authorization-server issuer equal to the normalized frontend base URL followed by `/api/auth`. The public auth base URL SHALL take precedence over the API base URL when configured. It SHALL NOT advertise the bare frontend origin as the issuer.

#### Scenario: Public auth base differs from the internal API

- **WHEN** the API base is `http://auto-ontology-frontend:3000` and the public auth base is `http://localhost:18300`
- **THEN** the advertised authorization server is exactly `http://localhost:18300/api/auth`

#### Scenario: Default configuration has no public override

- **WHEN** the public auth base is unset and the API base is `https://ontology.example`
- **THEN** the advertised authorization server is exactly `https://ontology.example/api/auth`

#### Scenario: Configured base URL has trailing slashes

- **WHEN** the public auth base is `https://ontology.example///`
- **THEN** the advertised authorization server is exactly `https://ontology.example/api/auth`, without duplicate slashes or a trailing slash

### Requirement: Publish issuer-specific authorization-server discovery

The frontend SHALL serve public authorization-server metadata at `/.well-known/oauth-authorization-server/api/auth` without requiring sign-in. Its `issuer` SHALL exactly match the issuer advertised by MCP. The existing origin-level discovery alias SHALL remain available and return the same issuer and authorization endpoints.

#### Scenario: Standards-based issuer discovery

- **WHEN** a client fetches `http://localhost:18300/.well-known/oauth-authorization-server/api/auth`
- **THEN** the response is HTTP 200 JSON with issuer `http://localhost:18300/api/auth`
- **AND** the authorization and token endpoints remain under `http://localhost:18300/api/auth/oauth2/`

#### Scenario: Existing origin-level discovery remains available

- **WHEN** a client fetches `http://localhost:18300/.well-known/oauth-authorization-server`
- **THEN** the response remains HTTP 200 JSON with the same issuer, authorization endpoint, token endpoint, and registration endpoint as issuer-specific discovery

### Requirement: Preserve private routing and caller identity

Selecting the public issuer SHALL NOT change the internal API destination, UserInfo token-verification destination, or public MCP resource URL. MCP calls SHALL continue to use the signed-in caller's bearer identity; issuer compatibility SHALL NOT be achieved by disabling issuer checks or introducing shared credentials.

#### Scenario: Private requests stay private

- **WHEN** public auth and private API base URLs differ
- **THEN** UserInfo verification and tool API calls use the private API base URL
- **AND** their bearer identity remains the signed-in caller's token

#### Scenario: Resource URL remains independent of issuer

- **WHEN** the MCP public origin is `http://localhost:18303`
- **THEN** the resource remains exactly `http://localhost:18303/mcp`, independently of the frontend issuer's `/api/auth` path
