// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, it, vi } from 'vitest';
import { conversationsApi, toConversation, type ConversationDetail } from './conversations';

afterEach(() => vi.unstubAllGlobals());
it('maps binding including legacy absence without discarding any messages', () => {
	const detail = {
		id: 'thread',
		title: 'Title',
		created_at: '2026-01-01',
		updated_at: '2026-01-01',
		connection_id: 'a',
		messages: ['user', 'assistant', 'assistant'].map((role, i) => ({
			id: `${i}`,
			conversation_id: 'thread',
			role,
			content: 'content',
			sql_code: null,
			sql_response: i === 2 ? 'chart rows' : null,
			created_at: '2026-01-01',
		})),
	} as ConversationDetail;
	expect(toConversation(detail)).toMatchObject({
		connectionId: 'a',
		messages: [
			{ role: 'user' },
			{ role: 'assistant' },
			{ role: 'assistant', sqlResponse: 'chart rows' },
		],
	});
	const { connection_id: _omitted, ...legacy } = detail;
	expect(_omitted).toBe('a');
	expect(toConversation(legacy as ConversationDetail).connectionId).toBeNull();
});
it('sends manual selectors in snake case while keeping old calls compatible', async () => {
	const fetcher = vi
		.fn()
		.mockImplementation(() => Promise.resolve(Response.json({ id: 'message' })));
	vi.stubGlobal('fetch', fetcher);
	for (const selector of [
		{ connectionId: 'a' },
		{ connectionLabel: 'Sales' },
		{ target_db: 'sales' },
		{},
	])
		await conversationsApi.addMessage('thread', { role: 'user', content: 'q', ...selector });
	expect(fetcher.mock.calls.map(([, init]) => JSON.parse(init.body))).toEqual([
		{ role: 'user', content: 'q', sql_code: null, sql_response: null, connection_id: 'a' },
		{
			role: 'user',
			content: 'q',
			sql_code: null,
			sql_response: null,
			connection_label: 'Sales',
		},
		{ role: 'user', content: 'q', sql_code: null, sql_response: null, target_db: 'sales' },
		{ role: 'user', content: 'q', sql_code: null, sql_response: null },
	]);
});
it('creates and renames with title only, never binding a source', async () => {
	const fetcher = vi.fn().mockImplementation(() => Promise.resolve(Response.json({})));
	vi.stubGlobal('fetch', fetcher);
	await conversationsApi.create('New');
	await conversationsApi.rename('thread', 'Renamed');
	expect(fetcher.mock.calls.map(([, init]) => [init.method, JSON.parse(init.body)])).toEqual([
		['POST', { title: 'New' }],
		['PATCH', { title: 'Renamed' }],
	]);
});
