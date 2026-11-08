// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { ConnectionType } from '@/enums/connection';
import type { Connection } from '@/types/connection';
import { ConnectionsInfoCardView } from './ConnectionsInfoCardView';

// Next's SVGR loader is not part of Vitest; only decorative icons are omitted.
vi.mock('@/common/icons', async (importOriginal) => ({
	...(await importOriginal<typeof import('@/common/icons')>()),
	Icon: () => null,
}));

const connections: Connection[] = ['Local', 'Production'].map((label, index) => ({
	connection_id: `id-${index}`,
	display_label: label,
	database_name: 'sales',
	connection: {
		type: ConnectionType.DATABRICKS,
		database: 'sales',
		host: 'host',
		http_path: '/sql',
		password: '',
	},
}));

describe('same-name connection cards (server rendering)', () => {
	it('shows human labels, physical names and ID-specific SSO controls independently', () => {
		const html = renderToStaticMarkup(
			createElement(ConnectionsInfoCardView, {
				connections,
				onEdit: vi.fn(),
				onDelete: vi.fn(),
				onSsoFederationChange: vi.fn(),
				ssoFederationPending: 'id-1',
			}),
		);
		expect(html).toContain('Actions for Local');
		expect(html).toContain('Actions for Production');
		expect(html.match(/>sales<\/p>/g)).toHaveLength(2);
		expect(html).toContain('data-testid="sso-federation-id-0"');
		expect(html).toContain('data-testid="sso-federation-id-1"');
		const inputs = html.match(/<input[^>]*>/g) ?? [];
		expect(inputs).toHaveLength(2);
		expect(inputs[0]).not.toContain(' disabled=""');
		expect(inputs[1]).toContain(' disabled=""');
		expect(html).not.toContain('Actions for id-');
	});
});
