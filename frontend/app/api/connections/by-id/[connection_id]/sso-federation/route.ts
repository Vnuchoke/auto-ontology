// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Toggle per-user SSO for this source ID without sending stored credentials.
export const PATCH = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));
