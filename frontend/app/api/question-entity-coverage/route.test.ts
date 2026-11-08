// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { permission, subjectToken } = vi.hoisted(() => ({
	permission: vi.fn(),
	subjectToken: vi.fn(),
}));
vi.mock('@/auth/with-auth', () => ({
	withPermission: (required: unknown) => {
		permission(required);
		return (handler: (req: NextRequest, ctx: unknown) => Promise<Response>) => handler;
	},
}));
vi.mock('@/auth/sso-token', () => ({ resolveSubjectToken: subjectToken }));

import { POST } from './route';

beforeEach(() => {
	subjectToken.mockReset();
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ data: {} })));
});
afterEach(() => vi.unstubAllGlobals());

test('keeps chat use permission', () => {
	expect(permission).toHaveBeenCalledWith({ chat: ['use'] });
});

test.each(['connection_id', 'connection_label', 'target_db'])(
	'forwards %s and the resolved SSO subject, not an unverified header',
	async (selector) => {
		subjectToken.mockResolvedValue('resolved-subject');
		const payload = { question: 'sales?', [selector]: 'East' };
		const req = new NextRequest('http://localhost/api/question-entity-coverage', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json', Authorization: 'Bearer incoming' },
			body: JSON.stringify(payload),
		});
		await POST(req, { user: { id: 'owner' } } as never);
		expect(subjectToken).toHaveBeenCalledWith(req.headers, 'owner');
		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toBe('http://127.0.0.1:3001/api/question-entity-coverage');
		expect(JSON.parse(init?.body as string)).toEqual(payload);
		expect(new Headers(init?.headers).get('authorization')).toBe('Bearer resolved-subject');
	},
);

test.each([401, 404, 409, 422, 503])(
	'preserves source refusal %i with no token fallback',
	async (status) => {
		subjectToken.mockResolvedValue(null);
		const error = { detail: { code: 'source_unavailable' } };
		vi.mocked(fetch).mockResolvedValue(Response.json(error, { status }));
		const req = new NextRequest('http://localhost/api/question-entity-coverage', {
			method: 'POST',
			headers: { Authorization: 'Bearer ignored' },
			body: JSON.stringify({ question: 'sales?', connection_id: 'selected' }),
		});
		const response = await POST(req, { user: { id: 'owner' } } as never);
		expect(response.status).toBe(status);
		expect(await response.json()).toEqual(error);
		expect(new Headers(vi.mocked(fetch).mock.calls[0][1]?.headers).has('authorization')).toBe(
			false,
		);
	},
);
