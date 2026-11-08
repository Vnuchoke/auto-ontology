// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { getChatSourceState } from './chatSourceState';

const a = { connection_id: 'a', display_label: 'A', database_name: 'db', dialect: 'postgresql' };
const b = { ...a, connection_id: 'b', display_label: 'B' };
const empty = { connectionId: null, hasMessages: false };

describe('chat source selection and immutable binding', () => {
	it.each([
		[[], null],
		[[a], 'a'],
		[[a, b], null],
	] as const)('selects automatically only for a sole source: %j', (sources, expected) => {
		const state = getChatSourceState(empty, null, [...sources], null);
		expect(state.selectedId).toBe(expected);
		expect(state.canSend).toBe(expected !== null);
	});

	it('keeps an explicit missing choice instead of falling back to its neighbor', () => {
		const state = getChatSourceState(empty, null, [b], 'a');
		expect(state.selectedId).toBe('a');
		expect(state.canSend).toBe(false);
	});

	it('locks persisted bindings, resolving labels by stable ID rather than database name', () => {
		const state = getChatSourceState({ connectionId: 'b', hasMessages: true }, a, [a, b], 'a');
		expect(state).toMatchObject({ mode: 'bound', connectionId: 'b', source: b, canSend: true });
	});

	it('accepts hook authority before the post-send detail is loaded', () => {
		expect(getChatSourceState(empty, b, [a, b], 'a')).toMatchObject({
			mode: 'bound',
			connectionId: 'b',
			source: b,
		});
	});

	it('treats any persisted message without a binding as legacy, not an empty draft', () => {
		expect(
			getChatSourceState({ connectionId: null, hasMessages: true }, null, [a], null),
		).toMatchObject({ mode: 'legacy', canSend: false });
	});

	it('locks unknown metadata even when a source choice remains', () => {
		expect(getChatSourceState(null, null, [a], 'a')).toMatchObject({
			mode: 'unknown',
			canSend: false,
		});
	});

	it('distinguishes discovery failure from confirmed removal and keeps only validated labels', () => {
		const bound = { connectionId: 'a', hasMessages: true };
		expect(getChatSourceState(bound, a, null, null)).toMatchObject({
			mode: 'bound',
			connectionId: 'a',
			source: a,
			canSend: false,
		});
		expect(getChatSourceState(bound, null, [], null)).toMatchObject({
			mode: 'unavailable',
			connectionId: 'a',
			source: null,
			canSend: false,
		});
	});
});
