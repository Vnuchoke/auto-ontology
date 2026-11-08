// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import { errorSchema, messageSchema } from '@/lib/openapiSchemas';
import type { OpenApiRoute } from '@/types/openapi';

const pathParams = z.object({
	id: z.string().describe('Conversation to append the message to. Must belong to the caller.'),
});

export const openapi: OpenApiRoute = {
	post: {
		path: pathParams,
		body: {
			description:
				'Chat turns are written by `POST /api/chat/completions` itself; this route is ' +
				'for callers that persist a turn they produced some other way. Source binding ' +
				'is immutable: bound conversations inherit their source, empty conversations bind ' +
				'on first append, and legacy unbound history is read-only. No model or connector runs.',
			schema: z.object({
				role: z.enum(['user', 'assistant']),
				content: z.string().optional(),
				sql_code: z.string().nullish(),
				sql_response: z.string().nullish(),
				connection_id: z
					.string()
					.nullish()
					.describe('Exact active source ID; use only one selector.'),
				connection_label: z
					.string()
					.nullish()
					.describe('Source display label; use only one selector.'),
				target_db: z
					.string()
					.nullish()
					.describe('Legacy source ID or unambiguous physical database name.'),
			}),
		},
		responses: {
			201: { description: 'The created message.', schema: messageSchema },
			404: {
				description: 'No such conversation, foreign owner, or selected source unavailable.',
				schema: errorSchema,
			},
			409: {
				description: 'Source conflict or legacy read-only history.',
				schema: errorSchema,
			},
			422: {
				description: 'Invalid, missing, or conflicting source selectors or message fields.',
				schema: errorSchema,
			},
			503: {
				description: 'Source metadata or conversation storage unavailable.',
				schema: errorSchema,
			},
		},
	},
};
