// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { isConnectionLabelTaken, proposeConnectionLabel } from './connectionLabels';

const connections = [
	{ connection_id: 'a', display_label: ' Local Sales ', database_name: 'sales' },
	{ connection_id: 'b', display_label: 'Production', database_name: 'sales' },
];

describe('connection labels', () => {
	it('compares trimmed labels without case, excluding only the edited ID', () => {
		expect(isConnectionLabelTaken(' local SALES ', connections)).toBe(true);
		expect(isConnectionLabelTaken('local sales', connections, 'a')).toBe(false);
		expect(isConnectionLabelTaken('PRODUCTION', connections, 'a')).toBe(true);
	});
	it('suggests the physical name whenever that label is free, without inventing a suffix', () => {
		expect(proposeConnectionLabel(' analytics ', connections)).toBe('analytics');
		expect(proposeConnectionLabel('sales', connections)).toBe('sales');
		expect(proposeConnectionLabel(' production ', connections)).toBe('');
		expect(proposeConnectionLabel('  ', connections)).toBe('');
	});
});
