// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withPermission } from '@/auth/with-auth';
import { proxyToBackend } from '@/auth/proxy-backend';

// Update settings or only the label of this stable source ID.
export const PUT = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));

// Retire only this source ID; an equal physical name does not identify a connection.
export const DELETE = withPermission({ connection: ['manage'] })((req) => proxyToBackend(req));
