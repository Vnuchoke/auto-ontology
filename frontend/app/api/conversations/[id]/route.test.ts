// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { beforeEach, expect, test, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { permission, db } = vi.hoisted(() => ({
	permission: vi.fn(),
	db: {
		conversation: {
			findFirst: vi.fn(),
			findUnique: vi.fn(),
			updateMany: vi.fn(),
			create: vi.fn(),
		},
	},
}));
vi.mock('@/auth/with-auth', () => ({
	withPermission: (required: unknown) => {
		permission(required);
		return (handler: (req: NextRequest, ctx: unknown) => Promise<Response>) => handler;
	},
}));
vi.mock('@/lib/prisma', () => ({ getPrisma: () => db }));

import { GET, PATCH } from './route';
import { POST } from '../route';

const row = {
	id: 'thread',
	user_id: 'owner',
	title: 'Title',
	connection_id: 'source-a' as string | null,
	created_at: '2026-11-08T00:00:00Z',
	updated_at: '2026-11-08T00:00:00Z',
	messages: [],
};
const context = () =>
	({ params: Promise.resolve({ id: 'thread' }), user: { id: 'owner' } }) as never;
const project = (value: typeof row, select: Record<string, unknown>) =>
	Object.fromEntries(Object.entries(value).filter(([key]) => select[key]));

beforeEach(() => {
	vi.clearAllMocks();
	db.conversation.findFirst.mockImplementation(async ({ select }) => project(row, select));
	db.conversation.findUnique.mockImplementation(async ({ select }) => project(row, select));
	db.conversation.updateMany.mockResolvedValue({ count: 1 });
	db.conversation.create.mockImplementation(async ({ select }) =>
		project({ ...row, connection_id: null }, select),
	);
});

test('owner-scoped detail exposes the saved CID needed before resume', async () => {
	const response = await GET(
		new NextRequest('http://localhost/api/conversations/thread'),
		context(),
	);
	expect(response.status).toBe(200);
	expect(await response.json()).toHaveProperty('connection_id', 'source-a');
	expect(db.conversation.findFirst.mock.calls[0][0].where).toEqual({
		id: 'thread',
		user_id: 'owner',
	});
	expect(db.conversation.findFirst.mock.calls[0][0].select.connection_id).toBe(true);
});

test('missing or foreign detail stays indistinguishable and does not reveal binding', async () => {
	db.conversation.findFirst.mockResolvedValue(null);
	const response = await GET(
		new NextRequest('http://localhost/api/conversations/thread'),
		context(),
	);
	expect(response.status).toBe(404);
	expect(await response.json()).toEqual({ error: 'Conversation not found' });
});

test('new conversations remain unbound even when a client sends a CID', async () => {
	const response = await POST(
		new NextRequest('http://localhost/api/conversations', {
			method: 'POST',
			body: JSON.stringify({ title: 'New', connection_id: 'source-b' }),
		}),
		context(),
	);
	expect(response.status).toBe(201);
	expect(await response.json()).toHaveProperty('connection_id', null);
	expect(db.conversation.create.mock.calls[0][0].data).toEqual({
		title: 'New',
		user_id: 'owner',
	});
});

test('conversation rename cannot rebind a source', async () => {
	const response = await PATCH(
		new NextRequest('http://localhost/api/conversations/thread', {
			method: 'PATCH',
			body: JSON.stringify({ title: 'Renamed', connection_id: 'source-b' }),
		}),
		context(),
	);
	expect(response.status).toBe(200);
	expect(await response.json()).toHaveProperty('connection_id', 'source-a');
	expect(db.conversation.updateMany.mock.calls[0][0]).toEqual({
		where: { id: 'thread', user_id: 'owner' },
		data: { title: 'Renamed' },
	});
});
