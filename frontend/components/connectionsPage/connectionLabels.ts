// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Connection } from '@/types/connection';

type ConnectionLabel = Pick<Connection, 'connection_id' | 'display_label'>;

// Provisional UI check; PostgreSQL remains authoritative, including concurrent edits.
export const isConnectionLabelTaken = (
	label: string,
	connections: readonly ConnectionLabel[],
	excludingConnectionId?: string,
): boolean =>
	connections.some(
		(connection) =>
			connection.connection_id !== excludingConnectionId &&
			connection.display_label.trim().toLowerCase() === label.trim().toLowerCase(),
	);

export const proposeConnectionLabel = (
	database: string,
	connections: readonly ConnectionLabel[],
): string => {
	const label = database.trim();
	return isConnectionLabelTaken(label, connections) ? '' : label;
};
