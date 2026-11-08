// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, expect, test, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { permission, prisma } = vi.hoisted(() => ({ permission: vi.fn(), prisma: vi.fn() }));
vi.mock('@/auth/with-auth', () => ({
	withPermission: (required: unknown) => {
		permission(required);
		return (handler: (req: NextRequest, ctx: unknown) => Promise<Response>) => handler;
	},
}));
vi.mock('@/lib/prisma', () => ({ getPrisma: prisma }));

import { POST } from './route';

beforeEach(() => {
	vi.stubGlobal('fetch', vi.fn());
	prisma.mockClear();
	prisma.mockImplementation(() => {
		throw new Error('Prisma must not write manual turns');
	});
});

test('keeps conversation write permission', () => {
	expect(permission).toHaveBeenCalledWith({ conversation: ['write'] });
});

test.each(['connection_id', 'connection_label', 'target_db'])(
	'forwards serialized %s and private identity, not bearer',
	async (selector) => {
		const message = {
			id: 'message',
			conversation_id: 'thread',
			role: 'user',
			content: 'q',
			sql_code: null,
			sql_response: null,
			created_at: '2026-11-08T00:00:00Z',
		};
		vi.mocked(fetch).mockResolvedValue(Response.json(message, { status: 201 }));
		const payload = { role: 'user', content: 'q', [selector]: 'East' };
		const req = new NextRequest('http://localhost/api/conversations/thread/messages', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				Authorization: 'Bearer secret',
				'x-auto-ontology-user-id': 'forged',
				'x-auto-ontology-source': 'mcp',
			},
			body: JSON.stringify(payload),
		});
		const response = await POST(req, {
			params: Promise.resolve({ id: 'thread' }),
			user: { id: 'owner' },
		} as never);
		expect(response.status).toBe(201);
		expect(await response.json()).toEqual(message);
		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toBe('http://127.0.0.1:3001/api/chat/conversations/thread/messages');
		expect(JSON.parse(init?.body as string)).toEqual(payload);
		const headers = new Headers(init?.headers);
		expect(headers.get('x-auto-ontology-user-id')).toBe('owner');
		expect(headers.get('x-auto-ontology-source')).toBe('mcp');
		expect(headers.has('authorization')).toBe(false);
		expect(prisma).not.toHaveBeenCalled();
	},
);

test.each([409, 404, 422, 503])(
	'preserves upstream source error %i without Prisma',
	async (status) => {
		const error = { detail: 'legacy_conversation_read_only' };
		vi.mocked(fetch).mockResolvedValue(Response.json(error, { status }));
		const req = new NextRequest('http://localhost/api/conversations/thread/messages', {
			method: 'POST',
			body: JSON.stringify({ role: 'user' }),
		});
		const response = await POST(req, {
			params: Promise.resolve({ id: 'thread' }),
			user: { id: 'owner' },
		} as never);
		expect(response.status).toBe(status);
		expect(await response.json()).toEqual(error);
		expect(prisma).not.toHaveBeenCalled();
	},
);
