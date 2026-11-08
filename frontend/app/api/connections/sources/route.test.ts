// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, test, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { permission, subjectToken } = vi.hoisted(() => ({
	permission: vi.fn(),
	subjectToken: vi.fn(),
}));
vi.mock('@/auth/with-auth', () => ({
	withPermission: (required: unknown) => {
		permission(required);
		return (handler: (req: NextRequest) => Promise<Response>) => handler;
	},
}));
vi.mock('@/auth/sso-token', () => ({ resolveSubjectToken: subjectToken }));

import { GET } from './route';

afterEach(() => vi.unstubAllGlobals());

test('discovery uses chat permission and never resolves or forwards credentials', async () => {
	const body = {
		data: [
			{
				connection_id: 'a',
				display_label: 'East',
				database_name: 'sales',
				dialect: 'postgresql',
			},
		],
		count: 1,
	};
	const fetcher = vi.fn().mockResolvedValue(Response.json(body));
	vi.stubGlobal('fetch', fetcher);
	const response = await GET(
		new NextRequest('http://localhost/api/connections/sources', {
			headers: { Authorization: 'Bearer incoming', 'x-auto-ontology-user-id': 'forged' },
		}),
		{} as never,
	);
	expect(permission).toHaveBeenCalledWith({ chat: ['use'] });
	expect(response.status).toBe(200);
	expect(await response.json()).toEqual(body);
	expect(response.headers.get('cache-control')).toBe('private, no-store');
	const [url, init] = fetcher.mock.calls[0];
	expect(url).toBe('http://127.0.0.1:3001/api/connections/sources');
	const headers = new Headers(init.headers);
	expect(headers.has('authorization')).toBe(false);
	expect(headers.has('x-auto-ontology-user-id')).toBe(false);
	expect(subjectToken).not.toHaveBeenCalled();
});
