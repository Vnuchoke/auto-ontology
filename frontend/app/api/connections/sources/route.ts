// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Discover active, credential-free query sources without settings access or driver probes.
export const GET = withPermission({ chat: ['use'] })((req) => proxyToBackend(req));
