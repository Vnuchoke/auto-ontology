// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod';
import type { OpenApiRoute } from '@/types/openapi';
import { sourceSelectors, sourceErrorSchema } from '@/lib/openapiSchemas';

// The handler forwards the parsed body to FastAPI verbatim, so the shape is the
// backend's `ChatRequest`. It is restated here rather than merged from
// `docs/openapi/backend.json` because this hand-rolled streaming proxy also
// enforces conversation permissions before forwarding the body.
export const openapi: OpenApiRoute = {
	post: {
		body: {
			description:
				'SQL plus a formatted answer, streamed as SSE. When the answer has an ' +
				'executed result, the backend also generates and persists a chart (or ' +
				'fallback table) and streams it back on this same connection as its own ' +
				'`charts` event — no follow-up call needed. Reuse one `conversation_id` for ' +
				'follow-up turns; an unknown UUID creates a conversation, while an ID owned ' +
				'by another user returns 404. Omitting `conversation_id` runs the question ' +
				'statelessly, with no history and no chart step. Select exactly one source ' +
				'by ID or unique label when multiple sources exist. Bound conversations ' +
				'inherit their immutable source when selectors are omitted. Legacy chats ' +
				'with messages but no source are read-only. SSE starts with a `source` ' +
				'snapshot {connection_id, display_label, database_name}, also in `answer.source`.',
			schema: z.object({
				question: z.string().min(1),
				evidence: z.string().nullish(),
				conversation_id: z.string().uuid().nullish(),
				...sourceSelectors,
				prediction: z.boolean().nullish(),
			}),
		},
		responses: {
			404: {
				description:
					'Source is unknown/inactive, or conversation is not owned by the caller.',
				schema: sourceErrorSchema,
			},
			409: {
				description:
					'Relayed from the backend: a run is already in flight for this ' +
					'conversation, source conflict, legacy read-only conversation, or ambiguous legacy name.',
				schema: sourceErrorSchema,
			},
			422: {
				description: 'Source selection is required, invalid, or conflicting.',
				schema: sourceErrorSchema,
			},
			503: {
				description: 'Selected source or its query projection is not ready.',
				schema: sourceErrorSchema,
			},
		},
	},
};
