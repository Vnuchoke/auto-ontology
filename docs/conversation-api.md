<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conversation API

Clients ask questions through the authenticated Next.js gateway:

```text
POST /api/chat/completions
```

Authenticate with a session cookie or an API token (see
[authentication.md](./authentication.md)). The request and response schemas are
in [`openapi/auto-ontology-api.json`](./openapi/auto-ontology-api.json).

## Select a source

`GET /api/connections/sources` requires `chat:use` and lists active source IDs,
unique labels, physical names and dialects without credentials or driver probes.
Pass exactly one of `connection_id` or `connection_label` when several sources
exist. The legacy `target_db` physical-name selector works only when unambiguous.
Omitting every selector auto-selects only a sole active source, or inherits a
bound conversation's saved source. Query readiness is checked for that source;
failure never switches to another connection or bypasses mandatory SSO.

In the web UI, a new chat with multiple sources has no default-first selection.
Choose its source before sending. A bound thread shows a locked source header;
start **New chat** to use another source. Legacy threads with messages but no CID
and threads whose source is inactive/deleted keep their history read-only.

## Response stream

The response is a server-sent event stream. Each event's `data:` line holds one
JSON object, and the stream ends with `data: [DONE]`, a final line that isn't
JSON.

| `type`   | Carries |
| -------- | ------- |
| `source` | The accepted snapshot `{connection_id, display_label, database_name}`, before model steps. |
| `step`   | An agent step starting or ending (`node`, `phase`, and `thought` on end), for showing progress. |
| `sql`    | A query that passed validation and is about to run. |
| `result` | The final `answer`: `response` (prose), `sql_code`, `sql_response_from_db` (rows), `thoughts` (step trace), and the same `source` snapshot. |
| `charts` | A chart or table for the result (`content`, `sql_response`). Sent after `result` when the query ran and the request includes a `conversation_id`. |
| `error`  | A `message` describing why the run failed. |

A run makes many sequential model calls and typically takes tens of seconds, so
read the stream as it arrives:

```python
import json
import os

import requests

response = requests.post(
    "https://auto-ontology.example.com/api/chat/completions",
    headers={"x-api-key": os.environ["AUTO_ONTOLOGY_API_TOKEN"]},
    json={"question": "How many orders shipped last week?"},
    stream=True,
    timeout=900,
)
response.raise_for_status()

for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        break
    event = json.loads(payload)
    if event["type"] == "step":
        print("…", event.get("node"), event.get("phase"))
    elif event["type"] == "result":
        answer = event["answer"]
        print(answer["response"])
        print(answer["sql_code"])
    elif event["type"] == "error":
        raise RuntimeError(event["message"])
```

The MCP server's `ask_question` tool wraps this same call and returns the
answer, SQL, and rows as separate fields; see [mcp.md](./mcp.md).
The example assumes a sole active source; add `connection_id` from discovery
to its JSON body for a multi-source deployment.

## Conversations

`conversation_id` is optional:

- Omit it for a stateless, one-shot request.
- Send a client-generated UUID to start or continue a conversation, and reuse it
  for every turn in the thread.
- Wait for `[DONE]` before sending the next turn. An overlapping request on the
  same conversation returns `409 Conversation in progress`.

```json
{
  "question": "What about last month?",
  "conversation_id": "d61d8aa3-d496-4fa7-97ce-4f831c162e7f"
}
```

For a conversational turn, the backend loads up to the five most recent
completed turns (capped at 12,000 characters) and rewrites a contextual
follow-up such as "What about August?" into a standalone question before
running the text-to-SQL agent. Independent questions pass through unchanged.
Only prior questions, answers, and SQL feed that rewrite; agent thoughts, chart
payloads, errors, and raw result rows stay out of it.

Conversation history belongs to the authenticated user. An API-created
conversation opens in the UI at `/chat?focus=<conversation_id>` for the user who
owns it. The web UI manages its own conversation IDs.

Conversation list/detail responses include nullable `connection_id`. Only the
backend's first accepted turn or manual append binds an empty thread; sending a
CID to conversation creation or title PATCH cannot rebind history. An explicit
different active source returns `409 source_conflict`. Existing messages with
NULL CID return `409 legacy_conversation_read_only`, even with a new selector.

Manual `POST /api/conversations/{id}/messages` uses the same atomic binding
protocol with `conversation:write`; it does not start a model, connector or SSO
exchange. Owner denial remains indistinguishable from a missing thread (404).

An HTTP refusal can occur after binding and persistence, for example when SSO
is required. The UI rereads detail instead of automatically repeating the
question. If that read also fails, it locks both sending and source selection
until **Retry recovery** succeeds. A discovery outage similarly preserves the
known history/CID and offers **Retry sources**; it is not evidence that a source
was deleted. Switching threads or starting a new chat discards late responses
from the previous selection.

## Trust boundary

Call the Next.js gateway (`APP_URL`, port 3000 in Docker Compose). It resolves
the user and forwards an internal `x-auto-ontology-user-id` header to FastAPI
(port 3001). FastAPI trusts that header without verifying it, so keep port 3001
on a private network. A direct FastAPI request without `conversation_id` stays
stateless; one with `conversation_id` requires the trusted identity header.
