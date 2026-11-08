// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

const originalAdapter = axios.defaults.adapter;
const sent: InternalAxiosRequestConfig[] = [];
let failed = false;
let sourcesApi: typeof import('./sources').sourcesApi;
const source = {
	connection_id: 'stable-id',
	display_label: 'Warehouse',
	database_name: 'physical_database',
	dialect: 'postgresql',
};
let availableSources = [source];

beforeAll(async () => {
	axios.defaults.adapter = async (config) => {
		sent.push(config);
		const response = {
			config,
			status: failed ? 503 : 200,
			statusText: failed ? 'Unavailable' : 'OK',
			headers: {},
			data: failed
				? { detail: 'Discovery unavailable' }
				: { data: availableSources, count: availableSources.length },
		};
		if (failed) throw new AxiosError('Unavailable', undefined, config, undefined, response);
		return response;
	};
	vi.resetModules();
	({ sourcesApi } = await import('./sources'));
});
beforeEach(() => {
	sent.length = 0;
	failed = false;
	availableSources = [source];
});
afterAll(() => {
	axios.defaults.adapter = originalAdapter;
	vi.resetModules();
});

describe('chat source discovery', () => {
	it('uses the permission-safe discovery GET without settings or source selectors', async () => {
		expect(await sourcesApi.list()).toEqual({ data: [source], count: 1 });
		expect(sent).toHaveLength(1);
		expect(sent[0]).toMatchObject({ method: 'get', url: 'connections/sources', params: {} });
		expect(sent[0].data).toBeUndefined();
	});

	it('returns discovery failure distinctly from an empty successful list', async () => {
		availableSources = [];
		expect(await sourcesApi.list()).toEqual({ data: [], count: 0 });
		failed = true;
		expect(await sourcesApi.list()).toEqual({ error: true, message: 'Discovery unavailable' });
	});
});
