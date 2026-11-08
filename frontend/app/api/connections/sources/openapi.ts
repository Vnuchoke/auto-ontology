// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { sourceDescriptorSchema, sourceErrorSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

export const openapi: OpenApiRoute = {
	get: {
		responses: {
			200: {
				description:
					'Active sources, without credentials or source-driver probes. Select one ID or unique label before querying when multiple sources exist.',
				schema: z.object({
					data: z.array(sourceDescriptorSchema.extend({ dialect: z.string() })),
					count: z.number().int(),
				}),
			},
			503: { description: 'Source metadata is not ready.', schema: sourceErrorSchema },
		},
	},
};
