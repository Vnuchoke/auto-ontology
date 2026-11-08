// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test } from 'vitest';
import spec from '../../docs/openapi/auto-ontology-api.json';

test('discovery is available with chat permission, never settings permission', () => {
	const paths = spec.paths as Record<string, Record<string, unknown>>;
	const operation = paths['/api/connections/sources']?.get;
	expect(operation).toBeDefined();
	expect(operation).toHaveProperty('x-auto-ontology-permissions', { chat: ['use'] });
	expect(JSON.stringify(operation)).not.toContain('credential_ref');
});

test('chat schema exposes both selectors to generated clients', () => {
	const properties =
		spec.paths['/api/chat/completions'].post.requestBody.content['application/json'].schema
			.properties;
	expect(properties).toHaveProperty('connection_id');
	expect(properties).toHaveProperty('connection_label');
});

test('management routes address IDs and retain settings permissions', () => {
	const paths = spec.paths as Record<string, Record<string, unknown>>;
	for (const [path, methods] of [
		['/api/connections/by-id/{connection_id}', ['put', 'delete']],
		['/api/connections/by-id/{connection_id}/sso-federation', ['patch']],
	] as const) {
		for (const method of methods) {
			expect(paths[path]?.[method]).toHaveProperty('x-auto-ontology-permissions', {
				connection: ['manage'],
			});
		}
	}
});
