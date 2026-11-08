// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, test, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { permission } = vi.hoisted(() => ({ permission: vi.fn() }));
vi.mock('@/auth/with-auth', () => ({
	withPermission: (required: unknown) => {
		permission(required);
		return (handler: (req: NextRequest) => Promise<Response>) => handler;
	},
}));

import { PUT, DELETE } from './route';
import { PATCH } from './sso-federation/route';

afterEach(() => vi.unstubAllGlobals());

test('all ID mutations retain connection manage permission', () => {
	expect(permission).toHaveBeenCalledTimes(3);
	for (const [required] of permission.mock.calls)
		expect(required).toEqual({ connection: ['manage'] });
});

test.each([
	['PUT', PUT, '', { display_label: 'East' }],
	['DELETE', DELETE, '', undefined],
	['PATCH', PATCH, '/sso-federation', { enabled: true }],
] as const)(
	'preserves %s ID, payload, errors and credential stripping',
	async (method, handler, suffix, payload) => {
		const error = { detail: 'source_not_found' };
		const fetcher = vi.fn().mockResolvedValue(Response.json(error, { status: 404 }));
		vi.stubGlobal('fetch', fetcher);
		const path = `/api/connections/by-id/source%20a${suffix}`;
		const response = await handler(
			new NextRequest(`http://localhost${path}`, {
				method,
				headers: { Authorization: 'Bearer secret', 'Content-Type': 'application/json' },
				body: payload ? JSON.stringify(payload) : undefined,
			}),
			{} as never,
		);
		expect(response.status).toBe(404);
		expect(await response.json()).toEqual(error);
		const [url, init] = fetcher.mock.calls[0];
		expect(url).toBe(`http://127.0.0.1:3001${path}`);
		expect(init.method).toBe(method);
		expect(init.body).toBe(payload ? JSON.stringify(payload) : '');
		expect(new Headers(init.headers).has('authorization')).toBe(false);
	},
);
