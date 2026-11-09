// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { memoryAdapter } from 'better-auth/adapters/memory';
import type { BetterAuthOptions } from 'better-auth';

const storage = vi.hoisted(() => ({
	available: false,
	error: Object.assign(new Error('the database system is starting up'), { code: '57P03' }),
	endpointError: undefined as Error | undefined,
	reads: vi.fn(),
	gate: undefined as Promise<void> | undefined,
	headers: vi.fn(async () => new Headers()),
}));

vi.mock('@/lib/prisma', () => ({ getPrisma: () => ({}) }));
vi.mock('better-auth/adapters/prisma', () => ({
	prismaAdapter: () => (options: BetterAuthOptions) => {
		const adapter = memoryAdapter({
			oauthResource: [{ id: 'resource', identifier: 'http://localhost:3003/mcp' }],
			oauthClient: [{ id: 'client', clientId: 'test-client', disabled: false }],
			oauthAccessToken: [
				{
					id: 'grant',
					token: createHash('sha256')
						.update('synthetic-opaque-token')
						.digest('base64url'),
					clientId: 'test-client',
					userId: 'test-user',
					scopes: ['openid', 'profile', 'email'],
					expiresAt: new Date('2099-01-01'),
					createdAt: new Date('2026-01-01'),
				},
			],
			user: [
				{
					id: 'test-user',
					name: 'Test User',
					email: 'user@example.invalid',
					emailVerified: true,
				},
			],
		})(options);
		const findOne = adapter.findOne.bind(adapter);
		adapter.findOne = async (query) => {
			storage.reads(query.model);
			if (query.model === 'oauthResource') await storage.gate;
			if (query.model === 'oauthResource' && !storage.available) throw storage.error;
			if (query.model === 'oauthAccessToken' && storage.endpointError)
				throw storage.endpointError;
			return findOne(query);
		};
		return adapter;
	},
}));
vi.mock('next/headers', () => ({
	headers: storage.headers,
	cookies: async () => ({ get: () => undefined, set: vi.fn() }),
}));

beforeEach(() => {
	vi.resetModules();
	storage.available = false;
	storage.gate = undefined;
	storage.endpointError = undefined;
	storage.reads.mockClear();
	storage.headers.mockReset().mockResolvedValue(new Headers());
	vi.stubEnv('AUTH_SECRET', 'synthetic-test-secret-at-least-thirty-two-characters');
	vi.stubEnv('APP_URL', 'http://localhost:3000');
	vi.stubEnv('AUTO_ONTOLOGY_MCP_PUBLIC_URL', 'http://localhost:3003');
	vi.stubEnv('BETTER_AUTH_TELEMETRY', 'false');
	vi.stubEnv('NEXT_PHASE', '');
	vi.stubGlobal(
		'fetch',
		vi.fn(() => {
			throw new Error('Network access forbidden');
		}),
	);
});

afterEach(() => {
	vi.unstubAllEnvs();
	vi.unstubAllGlobals();
});

const request = () =>
	new Request('http://localhost:3000/api/auth/oauth2/userinfo', {
		headers: { authorization: 'Bearer synthetic-opaque-token' },
	});
const context = { params: Promise.resolve({ all: ['oauth2', 'userinfo'] }) };

test('production UserInfo recovers after failed MCP initialization without reimporting', async () => {
	let release!: () => void;
	storage.gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	const { GET, POST } = await import('@/app/api/auth/[...all]/route');
	expect(POST).toBeTypeOf('function');
	const failed = expect(GET(request(), context)).rejects.toBe(storage.error);
	release();
	await failed;
	storage.available = true;
	const [recovered] = await Promise.allSettled([GET(request(), context)]);
	expect(recovered.status, 'healthy storage must recover on the next request').toBe('fulfilled');
	if (recovered.status !== 'fulfilled') throw recovered.reason;
	const response = recovered.value;
	expect(response.status).toBe(200);
	expect(await response.json()).toMatchObject({
		sub: 'test-user',
		email: 'user@example.invalid',
	});
});

test('import and blocked API-key routes never initialize auth', async () => {
	const { GET, POST } = await import('@/app/api/auth/[...all]/route');
	expect(storage.reads).not.toHaveBeenCalled();
	for (const handler of [GET, POST]) {
		const response = await handler(
			new Request('http://localhost:3000/api/auth/api-key/create'),
			context,
		);
		expect(response.status).toBe(404);
	}
	expect(storage.reads).not.toHaveBeenCalled();
});

test('concurrent failed and recovered waves each share one initialization, then retain success', async () => {
	const { getAuth } = await import('./auth');
	let release!: () => void;
	storage.gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	const first = getAuth();
	expect(getAuth()).toBe(first);
	const failure = expect(first).rejects.toBe(storage.error);
	await vi.waitFor(() => expect(storage.reads).toHaveBeenCalledTimes(1));
	let recovered!: ReturnType<typeof getAuth>;
	const nextWave = first.catch(() => {
		storage.available = true;
		recovered = getAuth();
		expect(getAuth()).toBe(recovered);
	});
	// This old waiter runs after the next wave starts and must not invalidate it.
	const lateWaiter = first.catch(() => {
		expect(getAuth()).toBe(recovered);
	});
	release();
	await Promise.all([failure, nextWave, lateWaiter]);
	const auth = await recovered;
	expect(await getAuth()).toBe(auth);
	expect(getAuth()).toBe(recovered);
	expect(storage.reads.mock.calls.filter(([model]) => model === 'oauthResource')).toHaveLength(2);
});

test('permanent initialization errors surface unchanged on each separate attempt', async () => {
	const { getAuth } = await import('./auth');
	const original = storage.error;
	storage.error = Object.assign(new Error('Invalid resource configuration'), {
		code: 'permanent',
	});
	try {
		await expect(getAuth()).rejects.toBe(storage.error);
		await expect(getAuth()).rejects.toBe(storage.error);
		expect(storage.reads).toHaveBeenCalledTimes(2);
	} finally {
		storage.error = original;
	}
});

test('a synchronous factory failure does not poison a later valid initialization', async () => {
	vi.stubEnv('AUTO_ONTOLOGY_MCP_PUBLIC_URL', 'invalid-relative-url');
	const { getAuth } = await import('./auth');
	await expect(getAuth()).rejects.toThrow();
	expect(storage.reads).not.toHaveBeenCalled();
	vi.stubEnv('AUTO_ONTOLOGY_MCP_PUBLIC_URL', 'http://localhost:3003');
	storage.available = true;
	const auth = await getAuth();
	expect(await getAuth()).toBe(auth);
	expect(storage.reads).toHaveBeenCalledTimes(1);
});

test('endpoint 401 and 500 after success do not reset initialization or replay operations', async () => {
	storage.available = true;
	const { getAuth } = await import('./auth');
	const { GET } = await import('@/app/api/auth/[...all]/route');
	const auth = await getAuth();
	const unauthorized = await GET(new Request(request().url), context);
	expect(unauthorized.status).toBe(401);
	expect(await getAuth()).toBe(auth);
	storage.endpointError = new Error('endpoint storage failure');
	const failure = await GET(request(), context);
	expect(failure.status).toBe(500);
	expect(await getAuth()).toBe(auth);
	expect(storage.reads.mock.calls.filter(([model]) => model === 'oauthAccessToken')).toHaveLength(
		1,
	);
	expect(storage.reads.mock.calls.filter(([model]) => model === 'oauthResource')).toHaveLength(1);
	storage.endpointError = undefined;
	expect((await GET(request(), context)).status).toBe(200);
});

test('Next request headers are resolved before auth initialization', async () => {
	const dynamicBoundary = new Error('Next dynamic headers boundary');
	storage.headers.mockRejectedValue(dynamicBoundary);
	const { getCurrentSession } = await import('./auth-guards');
	await expect(getCurrentSession()).rejects.toBe(dynamicBoundary);
	expect(storage.reads).not.toHaveBeenCalled();
});

test('discovery initializes inside the request and its issuer alias stays identical', async () => {
	const discovery = await import('@/app/.well-known/oauth-authorization-server/route');
	const alias = await import('@/app/.well-known/oauth-authorization-server/api/auth/route');
	expect(alias.GET).toBe(discovery.GET);
	expect(storage.reads).not.toHaveBeenCalled();
	await expect(
		discovery.GET(new Request('http://localhost:3000/.well-known/oauth-authorization-server')),
	).rejects.toBe(storage.error);
	storage.available = true;
	const response = await discovery.GET(
		new Request('http://localhost:3000/.well-known/oauth-authorization-server'),
	);
	expect(response.status).toBe(200);
	expect(await response.json()).toMatchObject({ userinfo_endpoint: request().url });
});
