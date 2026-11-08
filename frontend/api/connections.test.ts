// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConnectionType } from '@/enums/connection';
import type { ConnectionInput } from '@/types/connection';

const sent: InternalAxiosRequestConfig[] = [];
const originalAdapter = axios.defaults.adapter;
let conflict = false;
let connectionsApi: typeof import('./connections').connectionsApi;
const input: ConnectionInput = {
	type: ConnectionType.POSTGRESQL,
	database: 'sales',
	host: 'localhost',
	port: '5432',
	user: 'reader',
	password: '',
};

beforeAll(async () => {
	axios.defaults.adapter = async (config) => {
		sent.push(config);
		const response = {
			config,
			status: conflict ? 409 : 200,
			statusText: conflict ? 'Conflict' : 'OK',
			headers: {},
			data: conflict
				? { detail: 'This display label is already in use.' }
				: {
						success: true,
						schemas: ['public'],
						data: {
							connection_id: 'a',
							display_label: 'Local',
							database_name: 'sales',
							connection: input,
						},
					},
		};
		if (conflict)
			throw new AxiosError('Conflict', 'ERR_BAD_REQUEST', config, undefined, response);
		return response;
	};
	vi.resetModules();
	({ connectionsApi } = await import('./connections'));
});
beforeEach(() => {
	sent.length = 0;
	conflict = false;
});
afterAll(() => {
	axios.defaults.adapter = originalAdapter;
	vi.resetModules();
});

describe('connection management wire contract', () => {
	it('creates with a separate label, leaving physical database and blank secrets intact', async () => {
		const response = await connectionsApi.create(input, 'Local');
		expect(sent[0].url).toBe('connections');
		expect(JSON.parse(sent[0].data)).toEqual({ connection: input, display_label: 'Local' });
		expect(response).toMatchObject({
			data: { connection_id: 'a', display_label: 'Local', database_name: 'sales' },
		});
	});

	it('tests each same-name connection by ID, never the legacy replacing name', async () => {
		for (const id of ['a', 'b']) {
			expect(await connectionsApi.test(input, id)).toEqual({
				success: true,
				schemas: ['public'],
			});
		}
		expect(sent.map((request) => JSON.parse(request.data))).toEqual(
			['a', 'b'].map((id) => ({ connection: input, replacing_connection_id: id })),
		);
	});

	it('updates same-name records independently on encoded ID paths', async () => {
		for (const id of ['a', 'b/encoded']) await connectionsApi.update(id, input, `Label ${id}`);
		expect(sent.map((request) => request.url)).toEqual([
			'connections/by-id/a',
			'connections/by-id/b%2Fencoded',
		]);
		expect(sent.map((request) => JSON.parse(request.data))).toEqual(
			['a', 'b/encoded'].map((id) => ({ connection: input, display_label: `Label ${id}` })),
		);
	});

	it('renames without sending any driver settings', async () => {
		await connectionsApi.rename('b/encoded', 'Production');
		expect(sent[0].method).toBe('put');
		expect(sent[0].url).toBe('connections/by-id/b%2Fencoded');
		expect(JSON.parse(sent[0].data)).toEqual({ display_label: 'Production' });
	});

	it('toggles SSO and deletes only the requested ID', async () => {
		for (const id of ['a', 'b/encoded']) {
			await connectionsApi.setSsoFederation(id, true);
			await connectionsApi.delete(id);
		}
		expect(sent.map(({ method, url }) => [method, url])).toEqual([
			['patch', 'connections/by-id/a/sso-federation'],
			['delete', 'connections/by-id/a'],
			['patch', 'connections/by-id/b%2Fencoded/sso-federation'],
			['delete', 'connections/by-id/b%2Fencoded'],
		]);
		expect(JSON.parse(sent[0].data)).toEqual({ enabled: true });
	});

	it('preserves the authoritative server label-conflict message', async () => {
		conflict = true;
		expect(await connectionsApi.create(input, 'Local')).toEqual({
			error: true,
			message: 'This display label is already in use.',
		});
	});
});
