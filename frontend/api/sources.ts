// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { ApiResponse } from './types';
import type { SourceDescriptor } from '@/types/chat';

export type SourceSummary = SourceDescriptor & { dialect: string };

export const sourcesApi = {
	list: () => requests.get<ApiResponse<SourceSummary[]>>('connections/sources'),
};
