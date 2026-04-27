# Plan: Optimize tyagent Message Construction for Caching

## Goal

Maximize prompt caching (DeepSeek, Anthropic, etc.) by making the API request
prefix stable across turns. The system prompt + tool schemas + early messages
should be identical byte-by-byte between consecutive API calls.

## Current Issues

1. **Memory store injected as separate system message** (gateway.py:190-197):
   Each incoming message creates a new `{"role": "system", "content": memory_block}`
   dict inserted at a variable position. Changes prefix alignment.

2. **api_messages rebuilt every tool turn** (agent.py:115-127):
   `build_api_messages()` always returns a new list of new dicts, even when
   nothing changed. Breaks reference identity and adds copy overhead.

3. **system prompt not cached per session**:
   `TyAgent.__init__` builds the system prompt from `self.system_prompt` each
   time. No per-session stability guarantee.

4. **tool schemas in payload_base is stable but messages list isn't**:
   `tools=tool_defs` is same every time, but the prefix before user message
   keeps changing due to memory injection and copy overhead.

## Changes

### 1. Move memory into a stable cache block (gateway.py)

Instead of inserting a new system message every time, pre-build a once-per-session
memory block. Append it to the CURRENT user message content (like Hermes does),
not as a separate system message.

**Files to change:** `gateway.py` (~10 lines)

### 2. Make TyAgent system prompt stable per session (agent.py)

Cache the built system prompt on first call to `chat()`. Include the memory
block as part of the system prompt, not a separate message. Store it so that
subsequent `chat()` calls reuse the exact same string.

**Files to change:** `agent.py` (~15 lines)

### 3. Reuse `api_messages` by appending rather than rebuilding (agent.py)

In the tool loop, track the boundary between "already sent" messages and "new"
messages. Only build fresh `api_messages` when `should_compress()` triggers.
Otherwise, append only the latest assistant + tool messages to the previous
`api_messages` list.

**Files to change:** `agent.py` (~30 lines)

### 4. Build api_messages from mutable list only when needed (agent.py)

Remove the `build_api_messages` call from gateway.py's _on_message (line 187).
Let agent.chat() handle compression internally. Gateway's role is to provide
`session.messages` raw, not pre-compressed.

**Files to change:** `gateway.py` (~3 lines)

## Non-goals

- No Anthropic-style cache_control breakpoints (tyagent uses DeepSeek primarily)
- No change to tools schema delivery (already stable)
- No token counting change (already done in previous commit)
- No session DB changes
