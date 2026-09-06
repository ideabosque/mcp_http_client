# Agent Notes for mcp_http_client

## Project Basics

- Python project managed via `pyproject.toml` (Poetry backend).
- Package name: `mcp_http_client`.
- Python >=3.8.
- Dependencies: `aiohttp >=3.9.0`, `silvaengine_utility`.
- No CLI entrypoint — installed as a package and imported by `ai_agent_handler`, `mcp_daemon_engine`, and `mcp_proxy_engine`.
- No test runner config, no CI, no Makefile.

## Architecture

`mcp_http_client` is the async MCP JSON-RPC transport used across the SilvaEngine agent stack. It speaks the MCP Streamable HTTP transport (JSON-RPC 2.0 over HTTP POST), supports both plain-JSON and SSE responses, and exports tool definitions in the shape each LLM provider expects.

```
mcp_http_client/
  __init__.py            # Exports MCPHttpClient
  mcp_http_client.py     # MCPHttpClient — the full client
  models.py              # MCPTool, MCPResource, MCPPrompt dataclasses
  tests/
    test_mcp_http_client.py  # Manual smoke tests (skipped, requires live server + .env)
    test_deterministic.py    # Deterministic unit tests (no network)
```

### MCPHttpClient

- **Transport:** async, `aiohttp.ClientSession` with configurable total timeout (default 90s).
- **Protocol:** JSON-RPC 2.0 over HTTP POST.
- **Response modes:** plain JSON and SSE (`text/event-stream`) — auto-detected via `Content-Type`.
- **Auth:** optional Bearer token (`bearer_token` setting) or custom headers (`headers` setting).
- **MCP version:** `2024-11-05` (declared in `initialize`).
- **Timeout:** session-level via `setting["timeout"]` (default 90s), overridable per-call on `call_tool(timeout=...)`.

### Key methods

| Method | Purpose |
|--------|---------|
| `initialize()` | JSON-RPC `initialize` handshake (called on `__aenter__`) |
| `list_tools()` | `tools/list` → `List[MCPTool]` |
| `call_tool(name, arguments, timeout=None)` | `tools/call` → `List[Dict]` (content parts). Optional per-call timeout. |
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
| `gpt` | `{"type": "function", "name", "description", "parameters"}` (Responses API flat) |
| `ollama` | `{"type": "function", "function": {"name", "description", "parameters"}}` (Chat Completions nested) |
| `travrse` | `{"name", "description", "tool_type": "local", "parameters_schema"}` |

> `openai_completions_agent_handler._normalize_tools_to_chat_completions()` converts the flat `gpt` shape to nested Chat Completions. The `ollama` shape is already nested.

### Configuration (`setting` dict)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `base_url` | Yes | — | MCP server endpoint URL |
| `bearer_token` | No | `None` | JWT Bearer token |
| `headers` | No | `{}` | Custom headers |
| `timeout` | No | `90` | Total request timeout in seconds |

## Running tests

- **Deterministic unit tests** (no network):
  ```bash
  python -m pytest mcp_http_client/tests/test_deterministic.py -v
  ```
- **Manual smoke tests** require a `.env` file with `BASE_URL`, `X-API_KEY`:
  ```bash
  python -m mcp_http_client.tests.test_mcp_http_client
  ```

## Code Style

- Follow SilvaEngine conventions: `from __future__ import print_function` (or `annotations`), `__author__ = "bibow"` header, type hints.
- The client is async-only — all MCP operations use `aiohttp.ClientSession`.
- SSE parsing (`_parse_sse_response`) reads the full response body and extracts the last JSON-RPC payload from `data:` lines.

## Security Notes

- `bearer_token` is never logged by this client.
- `custom_headers` (including `x-api-key`) are merged into every request — callers must ensure no secrets leak into logs.
- The client does not validate `base_url` scheme — callers should ensure `https://` in production.
- The timeout prevents indefinite hangs when an MCP server is slow or stuck (e.g. a backend git-refresh triggered by `run_command`).