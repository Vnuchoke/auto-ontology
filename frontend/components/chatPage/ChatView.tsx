// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useChat } from '@/lib/useChat';
import { getChatSourceState } from '@/lib/chatSourceState';
import type { Conversation } from '@/types/chat';
import { conversationsApi, toConversation } from '@/api/conversations';
import { sourcesApi, type SourceSummary } from '@/api/sources';
import { ChatSidebar } from './ChatSidebar';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

export const ChatView = () => {
	const router = useRouter();
	const pathname = usePathname();
	const searchParams = useSearchParams();
	const focusId = searchParams.get('focus')?.trim() || null;
	const [activeConvId, setActiveConvId] = useState<string | null>(focusId);
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const [conversations, setConversations] = useState<Conversation[]>([]);
	const [sidebarLoading, setSidebarLoading] = useState(true);
	const [messageListLoading, setMessageListLoading] = useState(focusId !== null);
	const [persisted, setPersisted] = useState<{
		conversationId: string | null;
		connectionId: string | null;
		hasMessages: boolean;
	} | null>(focusId ? null : { conversationId: null, connectionId: null, hasMessages: false });
	const [chosenId, setChosenId] = useState<string | null>(null);
	const [sources, setSources] = useState<SourceSummary[] | null>(null);
	const [sourcesLoading, setSourcesLoading] = useState(true);
	const [sourcesError, setSourcesError] = useState<string | null>(null);
	const [sourcesRevision, setSourcesRevision] = useState(0);
	const [recoveryError, setRecoveryError] = useState<string | null>(null);
	const [accepting, setAccepting] = useState(false);
	const acceptingRef = useRef(false);
	const generationRef = useRef(0);
	const viewGeneration = generationRef.current;
	const activeIdRef = useRef<string | null>(focusId);
	const loadedFocusRef = useRef<string | null | undefined>(undefined);

	const {
		messages,
		setMessages,
		steps,
		liveSql,
		isLoading,
		sendMessage,
		resumeIfRunning,
		stopGeneration,
		clearConversation,
		source,
		error,
	} = useChat();
	const sourceState = getChatSourceState(
		persisted?.conversationId === activeConvId ? persisted : null,
		source,
		sources,
		chosenId,
	);
	const sourceRejected =
		error?.status === 404 &&
		(error.code === 'source_not_found' || error.code === 'source_unavailable');
	const busy = accepting || isLoading || messageListLoading;
	const canSend =
		sourceState.canSend && !sourcesLoading && !sourcesError && !sourceRejected && !busy;
	const canChoose =
		sourceState.mode === 'unbound' &&
		!busy &&
		!sourcesLoading &&
		!sourcesError &&
		!sourceRejected;

	const updateFocusInUrl = useCallback(
		(id: string | null) => {
			const params = new URLSearchParams(searchParams.toString());
			if (id) params.set('focus', id);
			else params.delete('focus');
			const query = params.toString();
			router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
		},
		[router, pathname, searchParams],
	);

	const refreshConversations = useCallback(async (generation = generationRef.current) => {
		try {
			const summaries = await conversationsApi.list();
			if (generation !== generationRef.current) return;
			setConversations(
				summaries.map((summary) => ({
					id: summary.id,
					title: summary.title || 'New conversation',
					connectionId: summary.connection_id,
					messages: [],
					createdAt: new Date(summary.created_at).getTime(),
				})),
			);
		} catch {
			// Keep the last successfully loaded sidebar.
		} finally {
			if (generation === generationRef.current) setSidebarLoading(false);
		}
	}, []);

	useEffect(() => {
		let active = true;
		setSourcesLoading(true);
		setSourcesError(null);
		sourcesApi
			.list()
			.then((res) => {
				if (!active) return;
				if (res.error) {
					setSources(null);
					setSourcesError(res.message || 'Could not load sources.');
				} else {
					setSources(res.data);
				}
			})
			.catch(() => {
				if (!active) return;
				setSources(null);
				setSourcesError('Could not load sources.');
			})
			.finally(() => {
				if (active) setSourcesLoading(false);
			});
		return () => {
			active = false;
		};
	}, [sourcesRevision]);

	useEffect(() => {
		if (sourceRejected) setSourcesRevision((revision) => revision + 1);
	}, [sourceRejected]);

	const loadDetail = useCallback(
		async (id: string, generation: number, resume: boolean) => {
			try {
				const detail = await conversationsApi.get(id);
				if (generation !== generationRef.current || activeIdRef.current !== id) return null;
				// Save binding metadata before watching; never infer legacy status from optimistic turns.
				setPersisted({
					conversationId: id,
					connectionId: detail.connection_id,
					hasMessages: detail.messages.length > 0,
				});
				setRecoveryError(null);
				setMessages(toConversation(detail).messages);
				if (resume) resumeIfRunning(id, detail.connection_id);
				return detail;
			} catch {
				if (generation !== generationRef.current || activeIdRef.current !== id) return null;
				setPersisted(null);
				setRecoveryError(
					'Could not verify this conversation’s source. Sending is locked until recovery succeeds.',
				);
				return null;
			} finally {
				if (generation === generationRef.current && activeIdRef.current === id)
					setMessageListLoading(false);
			}
		},
		[setMessages, resumeIfRunning],
	);

	const beginConversation = useCallback(
		(id: string | null) => {
			const generation = ++generationRef.current;
			activeIdRef.current = id;
			loadedFocusRef.current = id;
			acceptingRef.current = false;
			setAccepting(false);
			clearConversation();
			setActiveConvId(id);
			setChosenId(null);
			setRecoveryError(null);
			setPersisted(
				id ? null : { conversationId: null, connectionId: null, hasMessages: false },
			);
			setSidebarOpen(false);
			setMessageListLoading(id !== null);
			if (id) void loadDetail(id, generation, true);
			void refreshConversations(generation);
		},
		[clearConversation, loadDetail, refreshConversations],
	);

	useEffect(() => {
		if (loadedFocusRef.current !== focusId) beginConversation(focusId);
	}, [focusId, beginConversation]);

	useEffect(
		() => () => {
			generationRef.current += 1;
			loadedFocusRef.current = undefined;
		},
		[],
	);

	const handleNewChat = useCallback(() => {
		beginConversation(null);
		updateFocusInUrl(null);
	}, [beginConversation, updateFocusInUrl]);

	const handleSelectConversation = useCallback(
		(id: string) => {
			if (id !== activeIdRef.current) beginConversation(id);
			else setSidebarOpen(false);
			updateFocusInUrl(id);
		},
		[beginConversation, updateFocusInUrl],
	);

	const handleRename = useCallback(
		async (id: string, title: string) => {
			const generation = generationRef.current;
			try {
				await conversationsApi.rename(id, title);
				if (generation === generationRef.current) await refreshConversations(generation);
			} catch {
				/* Keep the existing title on failure. */
			}
		},
		[refreshConversations],
	);

	const handleDelete = useCallback(
		async (id: string) => {
			// Detach immediately so a late create/send/detail cannot resurrect the deleted selection.
			if (activeIdRef.current === id) handleNewChat();
			const generation = generationRef.current;
			try {
				await conversationsApi.delete(id);
				if (generation === generationRef.current) await refreshConversations(generation);
			} catch {
				/* Keep the sidebar entry if deletion fails. */
			}
		},
		[handleNewChat, refreshConversations],
	);

	const handleRecovery = useCallback(async () => {
		const id = activeIdRef.current;
		if (!id || acceptingRef.current) return;
		setMessageListLoading(true);
		await loadDetail(id, generationRef.current, true);
	}, [loadDetail]);

	const handleSend = useCallback(
		async (text: string): Promise<boolean> => {
			if (viewGeneration !== generationRef.current || activeConvId !== activeIdRef.current)
				return true;
			if (!canSend || acceptingRef.current || !sourceState.selectedId) return false;
			// Capture the target and lock re-entry before the first create await.
			const connectionId = sourceState.selectedId;
			const generation = generationRef.current;
			const previousMatchingTurns = messages.filter(
				(message) => message.role === 'user' && message.content === text,
			).length;
			acceptingRef.current = true;
			setAccepting(true);
			let id = activeIdRef.current;
			try {
				if (!id) {
					const created = await conversationsApi.create(
						text.slice(0, 50) || 'New conversation',
					);
					if (generation !== generationRef.current) return true;
					id = created.id;
					activeIdRef.current = id;
					loadedFocusRef.current = id;
					setActiveConvId(id);
					setPersisted({
						conversationId: id,
						connectionId: created.connection_id,
						hasMessages: false,
					});
					updateFocusInUrl(id);
					void refreshConversations(generation);
				}
				const accepted = await sendMessage(text, id, connectionId);
				if (generation !== generationRef.current || activeIdRef.current !== id) return true;
				if (accepted) return true;
				// Refusal can follow persistence (e.g. SSO/readiness): recover, never repeat the question.
				setPersisted(null);
				const detail = await loadDetail(id, generation, false);
				if (generation !== generationRef.current || activeIdRef.current !== id) return true;
				return (
					(detail?.messages.filter(
						(message) => message.role === 'user' && message.content === text,
					).length ?? 0) > previousMatchingTurns
				);
			} catch {
				if (generation !== generationRef.current) return true;
				if (id) {
					setPersisted(null);
					const detail = await loadDetail(id, generation, false);
					if (generation !== generationRef.current) return true;
					return (
						(detail?.messages.filter(
							(message) => message.role === 'user' && message.content === text,
						).length ?? 0) > previousMatchingTurns
					);
				}
				return false;
			} finally {
				if (generation === generationRef.current) {
					acceptingRef.current = false;
					setAccepting(false);
				}
			}
		},
		[
			viewGeneration,
			activeConvId,
			messages,
			canSend,
			sourceState.selectedId,
			sendMessage,
			loadDetail,
			refreshConversations,
			updateFocusInUrl,
		],
	);

	return (
		<div className="flex h-full bg-white dark:bg-zinc-950">
			<ChatSidebar
				conversations={conversations}
				activeId={activeConvId}
				onSelect={handleSelectConversation}
				onNewChat={handleNewChat}
				onRename={handleRename}
				onDelete={handleDelete}
				isOpen={sidebarOpen}
				onToggle={() => setSidebarOpen((open) => !open)}
				sidebarLoading={sidebarLoading}
			/>
			<main className="flex min-w-0 flex-1 flex-col">
				<div className="border-b border-zinc-200 px-4 py-3 text-sm dark:border-zinc-700">
					{sourceState.connectionId ? (
						<div>
							<p className="font-medium">
								Source: {sourceState.source?.display_label ?? 'Label unavailable'}
							</p>
							{sourceState.source && (
								<p className="text-secondary">{sourceState.source.database_name}</p>
							)}
							<p className="text-secondary">
								Connection ID: {sourceState.connectionId}
							</p>
							<p>
								This conversation’s source is locked. Start a new chat to switch
								sources.
							</p>
						</div>
					) : sourceState.mode === 'unbound' ? (
						<label className="flex flex-col gap-1">
							Source
							<select
								aria-label="Source"
								value={sourceState.selectedId ?? ''}
								disabled={!canChoose}
								onChange={(event) => {
									if (
										canChoose &&
										!acceptingRef.current &&
										viewGeneration === generationRef.current
									)
										setChosenId(event.target.value || null);
								}}
								className="rounded border border-zinc-300 bg-white p-2 dark:border-zinc-600 dark:bg-zinc-900"
							>
								<option value="">Choose a source</option>
								{chosenId &&
									!sources?.some((item) => item.connection_id === chosenId) && (
										<option value={chosenId}>
											Selected source unavailable
										</option>
									)}
								{sources?.map((item) => (
									<option key={item.connection_id} value={item.connection_id}>
										{item.display_label} — {item.database_name}
									</option>
								))}
							</select>
						</label>
					) : null}
					{sourcesLoading && <p role="status">Loading sources…</p>}
					{sourcesError && (
						<p role="alert">
							{sourcesError}{' '}
							<button
								type="button"
								onClick={() => setSourcesRevision((revision) => revision + 1)}
							>
								Retry sources
							</button>
						</p>
					)}
					{!sourcesLoading && !sourcesError && sources?.length === 0 && (
						<p>No sources available.</p>
					)}
					{sourceState.mode === 'legacy' && (
						<p role="status">
							This legacy conversation has no source binding and is read-only. Start a
							new chat to ask a question.
						</p>
					)}
					{(sourceState.mode === 'unavailable' || sourceRejected) && (
						<p role="alert">
							This conversation’s source is unavailable. History is read-only.
						</p>
					)}
					{recoveryError && (
						<p role="alert">
							{recoveryError}{' '}
							<button type="button" disabled={busy} onClick={handleRecovery}>
								Retry recovery
							</button>
						</p>
					)}
					{error && !sourceRejected && <p role="alert">{error.message}</p>}
					{(sourceState.mode !== 'unbound' || chosenId || sourceRejected) && (
						<button type="button" onClick={handleNewChat}>
							New chat
						</button>
					)}
				</div>
				<MessageList
					messages={messages}
					isLoading={isLoading}
					steps={steps}
					liveSql={liveSql}
					messageListLoading={messageListLoading}
				/>
				<ChatInput
					key={viewGeneration}
					onSend={handleSend}
					onStop={stopGeneration}
					isLoading={isLoading}
					disabled={!canSend}
				/>
			</main>
		</div>
	);
};
