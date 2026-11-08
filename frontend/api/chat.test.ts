// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from 'vitest';
import { streamChat, watchChat, type ChatEventCallbacks } from './chat';

const source = { connection_id: 'a', display_label: 'Sales', database_name: 'sales' };
const sse = (events: unknown[]) => {
	const text = events
		.map((event) => `data: ${typeof event === 'string' ? event : JSON.stringify(event)}\n\n`)
		.join('');
	return new Response(
		new ReadableStream({
			start(controller) {
				const bytes = new TextEncoder().encode(text);
				for (let i = 0; i < bytes.length; i += 7) controller.enqueue(bytes.slice(i, i + 7));
				controller.close();
			},
		}),
	);
};
const callbacks = () => ({
	onStep: vi.fn(),
	onSql: vi.fn(),
	onSource: vi.fn(),
	onResult: vi.fn(),
	onCharts: vi.fn(),
	onError: vi.fn(),
	onDone: vi.fn(),
	onStart: vi.fn(),
});
afterEach(() => vi.unstubAllGlobals());

describe('chat source wire boundary', () => {
	it.each(['connection_id', 'connection_label', 'target_db'] as const)(
		'serializes %s without other selectors',
		async (selector) => {
			const fetcher = vi.fn().mockResolvedValue(sse(['[DONE]']));
			vi.stubGlobal('fetch', fetcher);
			const cb = callbacks();
			streamChat({ question: 'q', conversationId: 'thread', [selector]: 'selected' }, cb);
			await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
			expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({
				question: 'q',
				conversation_id: 'thread',
				[selector]: 'selected',
			});
		},
	);
	it('omits selectors to inherit the persisted binding', async () => {
		const fetcher = vi.fn().mockResolvedValue(sse(['[DONE]']));
		vi.stubGlobal('fetch', fetcher);
		const cb = callbacks();
		streamChat({ question: 'q' }, cb);
		await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
		expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({ question: 'q' });
	});
	it.each([
		{ connection_id: 'a', target_db: 'db' },
		{ connection_label: ' ' },
		{ connection_id: 'a', connection_label: 'A' },
	])('rejects invalid selectors before fetch: %j', async (selectors) => {
		const fetcher = vi.fn();
		vi.stubGlobal('fetch', fetcher);
		const cb = callbacks();
		streamChat({ question: 'q', ...selectors }, cb);
		await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
		expect(fetcher).not.toHaveBeenCalled();
		expect(cb.onStart).not.toHaveBeenCalled();
		expect(cb.onError.mock.calls[0][0]).toMatchObject({
			code: 'invalid_source_selector',
			status: 422,
		});
	});
	it.each(['send', 'watch'])('preserves source, SQL, rows and charts for %s', async (mode) => {
		const answer = {
			response: 'answer',
			sql_code: 'select 1',
			sql_response_from_db: [{ n: '1' }],
			source,
		};
		const fetcher = vi
			.fn()
			.mockResolvedValue(
				sse([
					{ type: 'source', source: { ...source, password: 'secret' } },
					{ type: 'sql', sql: 'select 1' },
					{ type: 'result', answer },
					{ type: 'charts', content: 'chart' },
					'[DONE]',
				]),
			);
		vi.stubGlobal('fetch', fetcher);
		const cb = callbacks();
		if (mode === 'send') streamChat({ question: 'q', connection_id: 'a' }, cb);
		else watchChat('thread', { ...cb, expectedConnectionId: 'a' });
		await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
		expect(cb.onSource).toHaveBeenCalledWith({ type: 'source', source });
		expect(cb.onResult).toHaveBeenCalledWith({ type: 'result', answer });
		expect(cb.onSql).toHaveBeenCalledOnce();
		expect(cb.onCharts).toHaveBeenCalledOnce();
		expect(cb.onError).not.toHaveBeenCalled();
		if (mode === 'watch')
			expect(fetcher.mock.calls[0][0]).toBe('/api/chat/watch?conversation_id=thread');
	});
	it.each([
		null,
		{ ...source, display_label: ' ' },
		{ ...source, connection_id: 1 },
		{ ...source, connection_id: 'foreign' },
	])('blocks invalid or foreign result sources: %j', async (invalid) => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValue(
					sse([
						{ type: 'source', source },
						{ type: 'result', answer: { response: 'foreign secret', source: invalid } },
						{ type: 'charts', content: 'foreign chart' },
						'[DONE]',
					]),
				),
		);
		const cb = callbacks();
		streamChat({ question: 'q', connection_id: 'a' }, cb);
		await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
		expect(cb.onResult).not.toHaveBeenCalled();
		expect(cb.onCharts).not.toHaveBeenCalled();
		expect(JSON.stringify(cb.onError.mock.calls)).not.toContain('foreign secret');
	});
	it.each(['send', 'watch'])(
		'validates the first source and result-only sources for %s',
		async (mode) => {
			const fetcher = vi.fn();
			vi.stubGlobal('fetch', fetcher);
			for (const sourceEvent of [
				{ type: 'source', source: { ...source, connection_id: 'foreign' } },
				{ type: 'source', source: { ...source, database_name: '' } },
				{
					type: 'result',
					answer: {
						response: 'foreign answer',
						source: { ...source, connection_id: 'foreign' },
					},
				},
			]) {
				fetcher.mockResolvedValueOnce(sse([sourceEvent, '[DONE]']));
				const cb = callbacks();
				if (mode === 'send') streamChat({ question: 'q', connection_id: 'a' }, cb);
				else watchChat('thread', { ...cb, expectedConnectionId: 'a' });
				await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
				expect(cb.onSource).not.toHaveBeenCalled();
				expect(cb.onResult).not.toHaveBeenCalled();
			}
		},
	);
	it('sanitizes a valid result-only source without inventing an earlier source event', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(
				sse([
					{
						type: 'result',
						answer: { response: 'ok', source: { ...source, password: 'secret' } },
					},
					'[DONE]',
				]),
			),
		);
		const cb = callbacks();
		streamChat({ question: 'q', connection_id: 'a' }, cb);
		await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
		expect(cb.onResult).toHaveBeenCalledWith({
			type: 'result',
			answer: { response: 'ok', source },
		});
		expect(cb.onSource).not.toHaveBeenCalled();
		expect(cb.onStart).toHaveBeenCalledOnce();
	});
	it('preserves HTTP string detail and forwards structured watch errors', async () => {
		const fetcher = vi
			.fn()
			.mockResolvedValueOnce(Response.json({ detail: 'Choose a source.' }, { status: 422 }))
			.mockResolvedValueOnce(
				Response.json({ detail: { code: 'source_conflict' } }, { status: 409 }),
			);
		vi.stubGlobal('fetch', fetcher);
		const send = callbacks();
		const watch = callbacks();
		streamChat({ question: 'q' }, send);
		watchChat('thread', watch);
		await vi.waitFor(() => expect(send.onError).toHaveBeenCalledOnce());
		await vi.waitFor(() => expect(watch.onError).toHaveBeenCalledOnce());
		expect(send.onError).toHaveBeenCalledWith({
			type: 'error',
			message: 'Choose a source.',
			status: 422,
		});
		expect(watch.onError.mock.calls[0][0]).toMatchObject({
			code: 'source_conflict',
			status: 409,
		});
		expect(watch.onDone).not.toHaveBeenCalled();
	});
	it.each([
		[404, 'source_unavailable'],
		[409, 'source_conflict'],
		[409, 'legacy_conversation_read_only'],
		[422, 'source_required'],
		[503, 'source_not_ready'],
	] as const)('recognizes the backend chat string code for HTTP %s', async (status, code) => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(Response.json({ detail: code }, { status })),
		);
		const cb = callbacks();
		streamChat({ question: 'q' }, cb);
		await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
		expect(cb.onError.mock.calls[0][0]).toMatchObject({ status, code });
		expect(cb.onStart).not.toHaveBeenCalled();
	});
	it('rejects rebinding in repeated source events', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValue(
					sse([
						{ type: 'source', source },
						{ type: 'source', source: { ...source, connection_id: 'b' } },
						{ type: 'result', answer: { response: 'wrong' } },
						'[DONE]',
					]),
				),
		);
		const cb = callbacks();
		watchChat('thread', cb);
		await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
		expect(cb.onSource).toHaveBeenCalledTimes(1);
		expect(cb.onResult).not.toHaveBeenCalled();
	});
	it('allows older source-less streams and skips malformed progress lines', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValue(
					sse(['{bad json', { type: 'result', answer: { response: 'old' } }, '[DONE]']),
				),
		);
		const cb = callbacks();
		streamChat({ question: 'q', connection_id: 'a' }, cb);
		await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
		expect(cb.onResult).toHaveBeenCalledOnce();
		expect(cb.onSource).not.toHaveBeenCalled();
		expect(cb.onError).not.toHaveBeenCalled();
	});
	it.each([422, 404, 409, 503, 401])(
		'normalizes HTTP %s without accepting the turn',
		async (status) => {
			vi.stubGlobal(
				'fetch',
				vi
					.fn()
					.mockResolvedValue(
						Response.json(
							{ detail: { code: 'source_unavailable', password: 'secret' } },
							{ status },
						),
					),
			);
			const cb = callbacks();
			streamChat({ question: 'q' }, cb);
			await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
			expect(cb.onStart).not.toHaveBeenCalled();
			expect(cb.onError.mock.calls[0][0]).toMatchObject({
				type: 'error',
				code: 'source_unavailable',
				status,
			});
			expect(JSON.stringify(cb.onError.mock.calls)).not.toContain('secret');
		},
	);
	it('does not swallow callback exceptions as malformed JSON', async () => {
		vi.stubGlobal(
			'fetch',
			vi
				.fn()
				.mockResolvedValue(sse([{ type: 'result', answer: { response: 'ok' } }, '[DONE]'])),
		);
		const cb = callbacks();
		cb.onResult.mockImplementation(() => {
			throw new Error('callback failed');
		});
		streamChat({ question: 'q' }, cb);
		await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledOnce());
		expect(cb.onDone).not.toHaveBeenCalled();
	});
	it('keeps idle watch HTTP failures quiet except structured source errors', async () => {
		const cb: ChatEventCallbacks = callbacks();
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(new Response('unavailable', { status: 503 })),
		);
		watchChat('thread', cb);
		await vi.waitFor(() => expect(cb.onDone).toHaveBeenCalledOnce());
		expect(cb.onError).not.toHaveBeenCalled();
	});
});
