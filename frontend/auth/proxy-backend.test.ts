// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, test, vi } from 'vitest';
import { proxyToBackend } from './proxy-backend';

afterEach(() => vi.unstubAllGlobals());

test.each([undefined, 'mcp'])('source forwarding is opt-in: %s', async (source) => {
	const fetcher = vi.fn().mockResolvedValue(Response.json({ ok: true }));
	vi.stubGlobal('fetch', fetcher);
	await proxyToBackend(
		new Request('http://localhost/api/tags', {
			headers: {
				'x-auto-ontology-source': 'incoming',
				'x-auto-ontology-user-id': 'forged',
				Authorization: 'Bearer secret',
			},
		}),
		{ userId: 'owner', source },
	);
	const headers = new Headers(fetcher.mock.calls[0][1].headers);
	expect(headers.get('x-auto-ontology-source')).toBe(source ?? null);
	expect(headers.get('x-auto-ontology-user-id')).toBe('owner');
	expect(headers.has('authorization')).toBe(false);
});

test('only an explicitly resolved subject token can be forwarded', async () => {
	const fetcher = vi.fn().mockResolvedValue(Response.json({ ok: true }));
	vi.stubGlobal('fetch', fetcher);
	await proxyToBackend(
		new Request('http://localhost/api/question-entity-coverage', {
			method: 'POST',
			headers: { Authorization: 'Bearer untrusted' },
			body: '{}',
		}),
		{ subjectToken: 'resolved' },
	);
	expect(new Headers(fetcher.mock.calls[0][1].headers).get('authorization')).toBe(
		'Bearer resolved',
	);
});
