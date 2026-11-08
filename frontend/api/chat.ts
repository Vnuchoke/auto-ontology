// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
	ChatRequest,
	ChatStreamEvent,
	StepEvent,
	SqlEvent,
	ResultEvent,
	ChartsEvent,
	ErrorEvent,
	SourceDescriptor,
	SourceEvent,
} from '@/types/chat';

const getResponseError = async (res: Response): Promise<ErrorEvent> => {
	const error: ErrorEvent = {
		type: 'error',
		message: `Server responded with ${res.status}`,
		status: res.status,
	};
	const text = await res.text().catch(() => '');
	if (!text) return error;

	try {
		const { detail } = JSON.parse(text) as { detail?: unknown };
		if (typeof detail === 'string' && detail.trim()) error.message = detail;
		if (
			typeof detail === 'string' &&
			[
				'source_required',
				'invalid_source_selector',
				'ambiguous_source',
				'source_not_found',
				'source_unavailable',
				'source_not_ready',
				'source_conflict',
				'legacy_conversation_read_only',
			].includes(detail)
		)
			error.code = detail;
		if (
			detail &&
			typeof detail === 'object' &&
			'code' in detail &&
			typeof detail.code === 'string' &&
			detail.code.trim()
		) {
			error.code = detail.code;
			error.message =
				'The requested source is unavailable. Check the conversation source and try again.';
		}
	} catch {}
	return error;
};

const readSource = (value: unknown): SourceDescriptor | null => {
	if (!value || typeof value !== 'object') return null;
	const { connection_id, display_label, database_name } = value as Partial<SourceDescriptor>;
	if (
		typeof connection_id !== 'string' ||
		!connection_id.trim() ||
		typeof display_label !== 'string' ||
		!display_label.trim() ||
		typeof database_name !== 'string' ||
		!database_name.trim()
	)
		return null;
	return { connection_id, display_label, database_name };
};

export type ChatEventCallbacks = {
	expectedConnectionId?: string | null;
	onSource?: (event: SourceEvent) => void;
	onStep: (event: StepEvent) => void;
	/** The SQL the agent is about to run (or has just rewritten). */
	onSql?: (event: SqlEvent) => void;
	onResult: (event: ResultEvent) => void;
	onCharts?: (event: ChartsEvent) => void;
	onError: (event: ErrorEvent) => void;
	onDone?: () => void;
};

export type StreamChatCallbacks = ChatEventCallbacks & {
	/**
	 * Fires once the backend has accepted the request (HTTP 200 + body ready
	 * to stream). Useful for committing optimistic state — e.g. the user
	 * message — only after the server agreed to process it, so failures
	 * before the stream starts (e.g. 409 "Conversation in progress") don't
	 * leave orphan messages in the conversation.
	 */
	onStart?: () => void;
};

/**
 * Reads an already-open SSE `Response` body, dispatching `step`/`sql`/
 * `result`/`charts`/`error` events to `callbacks` as they arrive. Shared by
 * `streamChat` (submitting a new question) and `watchChat` (reattaching to a
 * run already in progress) since both consume the exact same wire format.
 *
 * Resolves once `[DONE]` is seen — which, unlike before, is now the *only*
 * normal terminator: `result` no longer ends the read early, because the
 * backend keeps the stream open a little longer to also generate and persist
 * the chart/table bubble (`charts`), and this needs to stay listening to
 * receive it. If the underlying connection closes before `[DONE]` ever
 * arrives — network drop, server restart, etc. — the caller has no way to
 * know how the run ended, so this surfaces it via `onError` rather than
 * leaving the UI stuck in a loading state forever.
 */
const consumeSseStream = async (res: Response, callbacks: ChatEventCallbacks): Promise<void> => {
	const reader = res.body!.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let streamCompleted = false;
	let connectionId = callbacks.expectedConnectionId ?? null;
	const acceptSource = (value: unknown): SourceDescriptor | null => {
		const source = readSource(value);
		if (!source || (connectionId !== null && source.connection_id !== connectionId)) {
			streamCompleted = true;
			callbacks.onError({
				type: 'error',
				code: 'source_mismatch',
				message: 'The response source could not be verified.',
			});
			return null;
		}
		connectionId = source.connection_id;
		return source;
	};

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;

		buffer += decoder.decode(value, { stream: true });
		const lines = buffer.split('\n');
		buffer = lines.pop() ?? '';

		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed.startsWith('data: ')) continue;

			const data = trimmed.slice(6);
			if (data === '[DONE]') {
				streamCompleted = true;
				break;
			}

			let event: ChatStreamEvent;
			try {
				event = JSON.parse(data);
			} catch {
				// Keep tolerating malformed progress lines, not callback/validation errors.
				continue;
			}
			if (!event || typeof event !== 'object') continue;
			switch (event.type) {
				case 'source': {
					const source = acceptSource(event.source);
					if (source) callbacks.onSource?.({ type: 'source', source });
					break;
				}
				case 'step':
					callbacks.onStep(event);
					break;
				case 'sql':
					callbacks.onSql?.(event);
					break;
				case 'result':
					if (event.answer && Object.hasOwn(event.answer, 'source')) {
						const source = acceptSource(event.answer.source);
						if (!source) break;
						event = { ...event, answer: { ...event.answer, source } };
					}
					callbacks.onResult(event);
					break;
				case 'charts':
					callbacks.onCharts?.(event);
					break;
				case 'error':
					streamCompleted = true;
					callbacks.onError(event);
					break;
			}
			if (streamCompleted) break;
		}
		if (streamCompleted) break;
	}

	// Stream closed without [DONE]/result/error — surface as an error so
	// the UI can drop out of loading instead of hanging forever.
	if (!streamCompleted) {
		callbacks.onError({
			type: 'error',
			message: 'Connection closed before the agent finished.',
		});
		return;
	}

	callbacks.onDone?.();
};

/**
 * Opens an SSE connection to POST /api/chat/completions and invokes
 * callbacks as events arrive. Returns an AbortController the caller
 * can use to cancel mid-stream.
 */
export const streamChat = (
	payload: ChatRequest,
	callbacks: StreamChatCallbacks,
): AbortController => {
	const controller = new AbortController();

	// The backend answers with the SQL/prose first (`result`), then generates
	// and persists the chart/table bubble itself and streams it back as a
	// separate `charts` event — no second request needed from here.
	const body = JSON.stringify({
		question: payload.question,
		conversation_id: payload.conversationId ?? undefined,
		connection_id: payload.connection_id ?? undefined,
		connection_label: payload.connection_label ?? undefined,
		target_db: payload.target_db ?? undefined,
	});

	(async () => {
		try {
			const selectors = [
				payload.connection_id,
				payload.connection_label,
				payload.target_db,
			].filter((value) => value != null);
			if (
				selectors.length > 1 ||
				selectors.some((value) => typeof value !== 'string' || !value.trim())
			) {
				// Defer so hook callers have installed their controller before callbacks fire.
				await Promise.resolve();
				if (!controller.signal.aborted)
					callbacks.onError({
						type: 'error',
						code: 'invalid_source_selector',
						status: 422,
						message: 'Choose exactly one valid source selector.',
					});
				return;
			}
			const res = await fetch('/api/chat/completions', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json', 'x-auto-ontology-source': 'app' },
				body,
				signal: controller.signal,
			});

			if (!res.ok || !res.body) {
				callbacks.onError(await getResponseError(res));
				return;
			}

			callbacks.onStart?.();
			await consumeSseStream(res, {
				...callbacks,
				expectedConnectionId: payload.connection_id ?? callbacks.expectedConnectionId,
			});
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') return;
			callbacks.onError({
				type: 'error',
				message: err instanceof Error ? err.message : 'Unknown error',
			});
		}
	})();

	return controller;
};

/**
 * Reattaches to a run already in progress for `conversationId`, if any —
 * e.g. after a page reload or reopening the browser mid-response. Replays
 * every step buffered server-side since the run started, then keeps
 * tailing live updates the same way `streamChat` does.
 *
 * If nothing is running for this conversation, the connection closes
 * immediately with `[DONE]` and `onDone` fires with no step/result/error
 * events — callers should treat that as a silent no-op, not an error.
 */
export const watchChat = (
	conversationId: string,
	callbacks: ChatEventCallbacks,
): AbortController => {
	const controller = new AbortController();

	(async () => {
		try {
			const res = await fetch(
				`/api/chat/watch?conversation_id=${encodeURIComponent(conversationId)}`,
				{ signal: controller.signal },
			);

			if (!res.ok || !res.body) {
				const error = await getResponseError(res);
				if (error.code) callbacks.onError(error);
				else callbacks.onDone?.();
				return;
			}

			await consumeSseStream(res, callbacks);
		} catch (err: unknown) {
			if (err instanceof DOMException && err.name === 'AbortError') return;
			callbacks.onError({
				type: 'error',
				message: err instanceof Error ? err.message : 'Unknown error',
			});
		}
	})();

	return controller;
};

/**
 * Aborts the run in flight for `conversationId`, freeing the conversation
 * so the next question isn't rejected with 409. Best-effort: aborting the
 * browser's own stream is what the user sees, and a failed cancel only
 * means the agent finishes on its own.
 */
export const cancelChat = async (conversationId: string): Promise<void> => {
	await fetch(`/api/chat/cancel?conversation_id=${encodeURIComponent(conversationId)}`, {
		method: 'POST',
	});
};
