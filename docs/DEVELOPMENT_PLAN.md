# MCP HTTP Client — Development Plan

> **Location:** `C:\Users\bibo7\gitrepo\silvaengine\mcp_http_client`
> **Date:** 2026-09-06
> **Status:** v0.0.1 — functional async MCP JSON-RPC client; no request timeout, no deterministic tests, no project docs

---

## Overview

`mcp_http_client` is the async HTTP client used by `ai_agent_handler` (and through it, `openai_completions_agent_handler`) to communicate with MCP servers over the MCP Streamable HTTP transport. It speaks JSON-RPC 2.0, supports both plain-JSON and SSE responses, and exports tool definitions in the shape each LLM provider expects.

It is the transport layer in the SilvaEngine agent stack:

```
openai_completions_agent_handler
  → ai_agent_handler (base class)
      → mcp_http_client          ← this project (MCP JSON-RPC over HTTP)
          → mcp_daemon_engine       (MCP server)
              → mcp_skill_provider    (in-process module)
                  → harness_engineering_engine  (GraphQL backend)
```

---

## Architecture

```
mcp_http_client/
  __init__.py            # Exports MCPHttpClient
  mcp_http_client.py     # MCPHttpClient — the full client
  models.py              # MCPTool, MCPResource, MCPPrompt dataclasses
  tests/
    test_mcp_http_client.py  # Manual smoke tests (skipped, requires live server + .env)
```

### MCPHttpClient

- **Transport:** async, `aiohttp.ClientSession`
- **Protocol:** JSON-RPC 2.0 over HTTP POST
- **Response modes:** plain JSON and SSE (`text/event-stream`) — auto-detected via `Content-Type`
- **Auth:** optional Bearer token (`bearer_token` setting) or custom headers (`headers` setting)
- **MCP version:** `2024-11-05` (declared in `initialize`)

### Key methods

| Method | Purpose |
|--------|---------|
| `initialize()` | JSON-RPC `initialize` handshake (called on `__aenter__`) |
| `list_tools()` | `tools/list` → `List[MCPTool]` |
| `call_tool(name, arguments)` | `tools/call` → `List[Dict]` (content parts) |
| `export_tools_for_llm(llm_name, tools)` | Transform `MCPTool` list into the tool-schema shape each LLM expects |
| `list_resources()` / `read_resource(uri)` | `resources/list`, `resources/read` |
| `list_prompts()` / `get_prompt(name, arguments)` | `prompts/list`, `prompts/get` |
| `health_check()` | `GET {base_url}/health` |
| `get_server_info()` | `GET {base_url}/` |

### `export_tools_for_llm` shapes

| `llm_name` | Output shape |
|------------|-------------|
| `gemini` | `{"name", "description", "parameters"}` |
| `claude` | `{"name", "description", "input_schema"}` |
| `gpt` | `{"type": "function", "name", "description", "parameters"}` (Responses API flat shape) |
| `ollama` | `{"type": "function", "function": {"name", "description", "parameters"}}` (Chat Completions nested shape) |
| `travrse` | `{"name", "description", "tool_type": "local", "parameters_schema"}` |

> **Note:** `openai_completions_agent_handler._normalize_tools_to_chat_completions()` converts the flat `gpt` shape into the nested Chat Completions shape. The `ollama` shape is already nested.

---

## Configuration

### `setting` dict (passed to `MCPHttpClient(logger, **setting)`)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `base_url` | Yes | — | MCP server endpoint URL (trailing `/` stripped) |
| `bearer_token` | No | `None` | JWT Bearer token sent as `Authorization: Bearer ...` |
| `headers` | No | `{}` | Custom headers merged into every request |
| `timeout` | No | *(none — see G-1)* | **Not yet implemented.** Should be the total request timeout in seconds |

### Where `setting` comes from

- **`ai_agent_handler`:** `agent["mcp_servers"][i]["setting"]` — each entry in the agent's `mcp_servers` list becomes one `MCPHttpClient` instance.
- **`mcp_daemon_engine`:** external MCP server settings stored in the `MCPSetting` row, injected at tool-execution time.
- **`mcp_skill_provider` tests:** constructed manually with an explicit `setting` dict.

---

## Known Gaps

Verified against the source on 2026-09-06:

### G-1 (Blocker) — No request timeout

`MCPHttpClient.__aenter__` creates `aiohttp.ClientSession()` with no `timeout` argument. Every `_send_request` POST inherits aiohttp's default (5 minutes total, unlimited connect). If an MCP server is slow or stuck — e.g. `run_command` triggers a slow git-refresh in `harness_engineering_engine` — the `mcp_http_client` → `mcp_daemon_engine` call hangs for minutes before failing.

This is the transport-layer root cause of the reliability gap tracked as item 7 in `mcp_skill_provider/docs/DEVELOPMENT_PLAN.md` and G-2 in `harness_engineering_engine/docs/DEVELOPMENT_PLAN.md` §18. Even after the backend gains a larger `runCommand` timeout budget, the client must also have its own deadline or it hangs indefinitely.

**Plan:**
- Accept `timeout` in `__init__` via `setting.get("timeout", 90)` (seconds, float).
- Pass `aiohttp.ClientTimeout(total=self._timeout)` to `ClientSession()`.
- Document `timeout` in the `setting` table above and in `README.md`.
- Allow per-call override via an optional `timeout` parameter on `call_tool` / `_send_request` for long-running operations.

### G-2 (Should fix) — No deterministic unit tests

All existing tests are `@unittest.skip` or require a live MCP server (`.env` with `BASE_URL`, `X-API_KEY`). No deterministic, network-free tests exist — unlike `openai_completions_agent_handler`, which has a `test_deterministic.py` suite.

The following methods are pure transformations that can be tested with zero network:

| Method | What to test |
|--------|-------------|
| `export_tools_for_llm` | Each of the 5 LLM shapes produces the correct output structure from a sample `MCPTool` list |
| `_parse_sse_response` | Correctly extracts JSON-RPC payload from multi-event SSE text, trailing event without blank line, malformed data |
| `_normalize_schema_keywords` | Integer/number/boolean keyword coercion, nested `properties` snake_case, `items`/`anyOf`/`oneOf` recursion |
| `_to_snake_case` | camelCase → snake_case conversion |
| `_clean_params` | None and empty-string removal, nested dict/list |

**Plan:** Add `tests/test_deterministic.py` covering all of the above with mocked inputs. Keep the existing `test_mcp_http_client.py` as the manual/live smoke suite.

### G-3 (Should fix) — README is empty

`README.md` contains only `# mcp_http_client` (17 bytes). Given this is the MCP transport used by every SilvaEngine agent, it should document:

- One-line description and role in the stack
- Install (`pip install -e .`)
- The `setting` dict contract (including the new `timeout` key from G-1)
- Async context manager usage pattern
- Method reference table (the one in this plan's Architecture section)
- `export_tools_for_llm` shape reference

**Plan:** Write `README.md` using the content structure from this plan's Architecture and Configuration sections.

### G-4 (Should fix) — No AGENTS.md

Every sibling project (`harness_engineering_engine`, `mcp_skill_provider`, `openai_completions_agent_handler`, `mcp_daemon_engine`) has an `AGENTS.md` with project basics, running instructions, architecture notes, code style, and testing guidance. `mcp_http_client` does not.

**Plan:** Create `AGENTS.md` covering:
- Project basics (Poetry, Python >=3.8, deps: `aiohttp`, `silvaengine_utility`)
- Running tests (deterministic suite + live smoke suite with `.env`)
- Architecture notes (transport, SSE parsing, schema normalization, LLM shape export)
- Code style (SilvaEngine conventions: `__future__` import, `__author__` header, type hints)
- The `export_tools_for_llm` shape table (critical for anyone wiring a new LLM handler)

### G-5 (Nice to have) — SSE parsing has no streaming support

`_parse_sse_response` reads the entire response body as text (`await response.text()`) and then extracts the last JSON object. This works for single-response JSON-RPC, but the MCP Streamable HTTP spec allows servers to stream multiple events. The current implementation discards all but the last payload.

**Plan:** If streaming tool results become needed (e.g. long-running `run_command` with progressive output from G-1 in the `harness_engineering_engine` plan), switch to `aiohttp`'s async line iterator (`response.content` / `async for line in response.content`) and yield events as they arrive. Not needed for v1 — defer until the backend has a streaming command path.

---

## Delivery Plan

### P1 — Add request timeout (G-1)

- Add `self._timeout = float(setting.get("timeout", 90))` to `__init__`
- Pass `timeout=aiohttp.ClientTimeout(total=self._timeout)` to `ClientSession()`
- Add optional `timeout` parameter to `call_tool()` and `_send_request()` for per-call override
- Update `setting` table in this plan and in `README.md` (P3)

**Exit criteria:** A hung MCP server connection fails within `timeout` seconds instead of hanging indefinitely.

### P2 — Add deterministic unit tests (G-2)

- Create `tests/test_deterministic.py`
- Cover `export_tools_for_llm` (all 5 shapes), `_parse_sse_response`, `_normalize_schema_keywords`, `_to_snake_case`, `_clean_params`
- No network dependencies, no `.env` required
- Add to `pyproject.toml` dev dependencies if needed (`pytest`)

**Exit criteria:** `python -m pytest tests/test_deterministic.py -v` passes with zero network calls.

### P3 — Write README (G-3)

- Install instructions
- `setting` contract (including `timeout`)
- Usage pattern (async context manager)
- Method reference
- `export_tools_for_llm` shape table

**Exit criteria:** A new developer can construct `MCPHttpClient` and call `list_tools` / `call_tool` from the README alone.

### P4 — Create AGENTS.md (G-4)

- Project basics, architecture notes, code style, testing guidance
- Reference the `export_tools_for_llm` shape table

**Exit criteria:** Consistent with sibling project `AGENTS.md` files.

---

## Dependencies

- `aiohttp >=3.9.0` — async HTTP client
- `silvaengine_utility` — `Debugger`, `convert_decimal_to_number`
- **Used by:**
  - `ai_agent_handler` — initializes `MCPHttpClient` instances from `agent["mcp_servers"]`
  - `ai_agent_core_engine` — `utils/mcp_tools.py` uses `MCPHttpClient` to list MCP server tools
  - `mcp_daemon_engine` — `mcp_external.py` uses `MCPHttpClient` for external MCP server inventory sync
  - `mcp_proxy_engine` — `function_handler.py` uses `MCPHttpClient` for tool execution

---

## Security Notes

- `bearer_token` is sent as `Authorization: Bearer ...` — never logged by this client.
- `custom_headers` (including `x-api-key`) are merged into every request — callers must ensure no secrets leak into logs.
- The client does not validate `base_url` scheme — callers should ensure `https://` in production.

---

## License

MIT