// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SourceSummary } from '@/api/sources';
import type { SourceDescriptor } from '@/types/chat';

// Only persisted detail determines legacy status; optimistic messages never do.
export const getChatSourceState = (
	persisted: { connectionId: string | null; hasMessages: boolean } | null,
	accepted: SourceDescriptor | null,
	sources: SourceSummary[] | null,
	chosenId: string | null,
) => {
	const connectionId = persisted?.connectionId ?? accepted?.connection_id ?? null;
	const selectedId =
		connectionId ?? chosenId ?? (sources?.length === 1 ? sources[0].connection_id : null);
	const listed = sources?.find((source) => source.connection_id === selectedId) ?? null;
	const source = listed ?? (accepted?.connection_id === connectionId ? accepted : null);
	const mode = !persisted
		? 'unknown'
		: persisted.connectionId === null && persisted.hasMessages
			? 'legacy'
			: connectionId
				? sources !== null && !listed
					? 'unavailable'
					: 'bound'
				: 'unbound';
	return {
		mode,
		connectionId,
		selectedId,
		source,
		canSend: (mode === 'bound' || mode === 'unbound') && listed !== null,
	};
};
