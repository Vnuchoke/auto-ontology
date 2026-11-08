// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { oauthProviderAuthServerMetadata } from '@better-auth/oauth-provider';
import { auth } from '@/auth/auth';

// Publishes Auto Ontology's OAuth authorization-server metadata (RFC 8414) so an MCP
// client can discover how to sign a user in.
//
// Keep the origin-level discovery alias; the /api/auth issuer's RFC 8414 route
// reuses this handler at /.well-known/oauth-authorization-server/api/auth.
// `/.well-known` is exempted from the page auth gate in proxy.ts.
export const GET = oauthProviderAuthServerMetadata(auth);
