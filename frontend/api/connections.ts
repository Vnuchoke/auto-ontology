// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { requests } from './requests';
import type { Connection } from '@/types/connection';
import type { ConnectionInput } from '@/types/connection';
import type { ApiError, ApiResponse, ResponseWithCount } from './types';

type CreateResponse = ApiError | { data: Connection };
// A successful test also returns the connection's schemas (for the picker).
type TestResponse = ApiError | { success: true; schemas: string[] };

const err = (message: string): ApiError => ({ error: true, message });

export const connectionsApi = {
	getAll: (): Promise<ApiResponse<Connection[]>> =>
		requests.get<ResponseWithCount<Connection[]>>('connections'),

	isEnvSource: () => requests.get<boolean>('connections/source'),

	create: (input: ConnectionInput, displayLabel: string): Promise<CreateResponse> =>
		requests.post<{ data: Connection }>('connections', {
			connection: input,
			display_label: displayLabel,
		}),

	test: async (input: ConnectionInput, replacingConnectionId?: string): Promise<TestResponse> => {
		const res = await requests.post<{ success: boolean; schemas?: string[] }>(
			'connections/test',
			{
				connection: input,
				...(replacingConnectionId == null
					? {}
					: { replacing_connection_id: replacingConnectionId }),
			},
		);

		if (res.error || res.success !== true) {
			return err(res.message ?? 'Connection test failed.');
		}

		// The test also reports the connection's schemas (empty for connectors
		// that don't enumerate them) so the UI can offer a schema picker.
		return { success: true, schemas: res.schemas ?? [] };
	},

	// Physical `database` and connector `type` remain immutable settings.
	// Credentials left blank keep their stored
	// value, since `getAll` strips them and the edit form has none to send back.
	update: (
		connectionId: string,
		input: ConnectionInput,
		displayLabel: string,
	): Promise<CreateResponse> =>
		requests.put<{ data: Connection }>(
			`connections/by-id/${encodeURIComponent(connectionId)}`,
			{
				connection: input,
				display_label: displayLabel,
			},
		),

	rename: (connectionId: string, displayLabel: string): Promise<CreateResponse> =>
		requests.put<{ data: Connection }>(
			`connections/by-id/${encodeURIComponent(connectionId)}`,
			{
				display_label: displayLabel,
			},
		),

	// Narrow update: flips "authenticate as signed-in user" without re-sending
	// credentials, so the connections page can toggle it inline.
	setSsoFederation: (connectionId: string, enabled: boolean) =>
		requests.patch<{
			data: {
				connection_id: string;
				display_label: string;
				database_name: string;
				sso_federation: boolean;
			};
		}>(`connections/by-id/${encodeURIComponent(connectionId)}/sso-federation`, { enabled }),

	delete: (connectionId: string) =>
		requests.delete<{
			data: { connection_id: string; display_label: string; database_name: string };
		}>(`connections/by-id/${encodeURIComponent(connectionId)}`),
};
