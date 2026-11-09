// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { headers } from 'next/headers';
import { redirect } from 'next/navigation';
import { getAuth } from '@/auth/auth';
import { Role } from '@/enums/auth';

/** Authoritative session lookup for server components and route handlers. */
export const getCurrentSession = async () => {
	const requestHeaders = await headers();
	return (await getAuth()).api.getSession({ headers: requestHeaders });
};

/** Require any authenticated user; redirect to /login otherwise. */
export const requireUser = async () => {
	const session = await getCurrentSession();
	if (!session) redirect('/login');
	return session;
};

/**
 * Require an admin. This is the real authorization gate — proxy.ts only does an
 * optimistic cookie-presence check and cannot read the role.
 */
export const requireAdmin = async () => {
	const session = await requireUser();
	if (session.user.role !== Role.Admin) redirect('/chat');
	return session;
};
