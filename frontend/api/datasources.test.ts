// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, expect, test, vi } from 'vitest';
import { datasources } from './datasources';
import { requests } from './requests';

afterEach(() => vi.restoreAllMocks());

test('same-name databases load isolated schemas and deduplicate only by root id', async () => {
	const get = vi.spyOn(requests, 'get').mockImplementation(async (path) => ({
		error: false,
		schemas_count: 1,
		schemas: [{ id: `${path}-schema`, schema_name: 'public', tables_count: 0 }],
	}));
	const first = datasources.getSchemasForDatabase('sales-a');
	const repeated = datasources.getSchemasForDatabase('sales-a');
	const second = datasources.getSchemasForDatabase('sales-b');
	expect(first).toBe(repeated);
	expect(first).not.toBe(second);
	const [a, b] = await Promise.all([first, second]);
	expect(get.mock.calls).toEqual([['schemas/sales-a'], ['schemas/sales-b']]);
	expect(a.data?.[0].id).toBe('schemas/sales-a-schema');
	expect(b.data?.[0].id).toBe('schemas/sales-b-schema');
	await datasources.getSchemasForDatabase('sales-a');
	expect(get).toHaveBeenCalledTimes(3);
});
