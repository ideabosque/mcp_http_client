# mcp_http_client

Async HTTP client for interacting with an MCP server using JSON-RPC 2.0 over the MCP Streamable HTTP transport.

Part of the SilvaEngine agent stack — used by `ai_agent_handler` (and through it, `openai_completions_agent_handler`) to call MCP servers. Also used by `mcp_daemon_engine` for external MCP server inventory sync, and by `mcp_proxy_engine` for tool execution.

## Install

```bash
pip install -e .
```

Requires `aiohttp >=3.9.0` and `silvaengine_utility`.

## Usage

```python
import logging
from mcp_http_client import MCPHttpClient

logger = logging.getLogger()

setting = {
    "base_url": "http://localhost:8000/mcp",
    # "bearer_token": "my-jwt-token",   # optional, sent as Authorization: Bearer
    # "headers": {"x-api-key": "..."},   # optional, custom headers
    "timeout": 90,                        # optional, total request timeout in seconds (default 90)
}

async with MCPHttpClient(logger, **setting) as client:
    # List available tools
    tools = await client.list_tools()

    # Export tools in the shape your LLM expects
    tools_for_llm = client.export_tools_for_llm("ollama", tools)

    # Call a tool (optional per-call timeout override)
    result = await client.call_tool("search_skills", {"query": "video"}, timeout=120)
```

## Configuration

The `setting` dict passed to `MCPHttpClient(logger, **setting)`:

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `base_url` | Yes | — | MCP server endpoint URL (trailing `/` stripped) |
| `bearer_token` | No | `None` | JWT Bearer token sent as `Authorization: Bearer ...` |
| `headers` | No | `{}` | Custom headers merged into every request |
| `timeout` | No | `90` | Total request timeout in seconds. Prevents indefinite hangs when the server is slow or stuck. Can be overridden per-call on `call_tool(timeout=...)`. |

## Methods

| Method | Purpose |
|--------|---------|
| `initialize()` | JSON-RPC `initialize` handshake (called automatically on `__aenter__`) |
| `list_tools()` | `tools/list` → `List[MCPTool]` |
| `call_tool(name, arguments, timeout=None)` | `tools/call` → `List[Dict]` (content parts). Optional `timeout` overrides the session-level timeout for this call only. |
| `export_tools_for_llm(llm_name, tools)` | Transform `MCPTool` list into the tool-schema shape each LLM expects |
| `list_resources()` | `resources/list` → `List[MCPResource]` |
| `read_resource(uri)` | `resources/read` → `Dict` |
| `list_prompts()` | `prompts/list` → `List[MCPPrompt]` |
| `get_prompt(name, arguments)` | `prompts/get` → `Dict` |
| `health_check()` | `GET {base_url}/health` → `Dict` |
| `get_server_info()` | `GET {base_url}/` → `Dict` |

## `export_tools_for_llm` output shapes

| `llm_name` | Output shape |
|------------|-------------|
| `gemini` | `{"name", "description", "parameters"}` |
| `claude` | `{"name", "description", "input_schema"}` |
| `gpt` | `{"type": "function", "name", "description", "parameters"}` (Responses API flat shape) |
| `ollama` | `{"type": "function", "function": {"name", "description", "parameters"}}` (Chat Completions nested shape) |
| `travrse` | `{"name", "description", "tool_type": "local", "parameters_schema"}` |

> **Note:** `openai_completions_agent_handler._normalize_tools_to_chat_completions()` converts the flat `gpt` shape into the nested Chat Completions shape. The `ollama` shape is already nested.

## Response handling

The client auto-detects the response format:
- **JSON** (`Content-Type: application/json`): parsed directly.
- **SSE** (`Content-Type: text/event-stream`): the JSON-RPC payload is extracted from `data:` lines via `_parse_sse_response`.
- **Incorrect content-type**: the client still attempts JSON parsing from the response text.

## Testing

### Deterministic tests (no network)

```bash
python -m pytest mcp_http_client/tests/test_deterministic.py -v
```

Covers `export_tools_for_llm` (all 5 LLM shapes), `_parse_sse_response`, `_normalize_schema_keywords`, `_to_snake_case`, and `_clean_params`.

### Live smoke tests (requires `.env`)

```bash
cp mcp_http_client/tests/.env.example .env
# Fill in BASE_URL, X-API_KEY
python -m mcp_http_client.tests.test_mcp_http_client
```

## Security

- `bearer_token` is sent as `Authorization: Bearer ...` — never logged by this client.
- `custom_headers` (including `x-api-key`) are merged into every request — callers must ensure no secrets leak into logs.
- The client does not validate `base_url` scheme — callers should ensure `https://` in production.

## License

MIT