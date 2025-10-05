#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from silvaengine_utility import convert_decimal_to_number

from .models import MCPPrompt, MCPResource, MCPTool


class MCPHttpClient:
    def __init__(self, logger: logging.Logger, **setting: List[Dict[str, Any]]):
        self.logger = logger
        self.base_url = setting["base_url"].rstrip("/")
        self.bearer_token = setting.get("bearer_token")
        self.custom_headers = setting.get("headers", {})
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id = 0
        self._initialized = False

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _clean_params(self, obj: Any) -> Any:
        """Recursively remove attributes that are None or empty strings."""
        if isinstance(obj, dict):
            return {
                k: self._clean_params(v) for k, v in obj.items() if v not in (None, "")
            }
        elif isinstance(obj, list):
            return [self._clean_params(item) for item in obj]
        return obj

    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
        }
        if params:
            cleaned_params = self._clean_params(params)
            if cleaned_params:
                request_data["params"] = convert_decimal_to_number(cleaned_params)

        headers = {
            "Content-Type": "application/json",
            **self.custom_headers,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        try:
            async with self.session.post(
                self.base_url,
                json=request_data,
                headers=headers,
            ) as response:
                response.raise_for_status()

                # Try to parse as JSON regardless of content-type header
                # Some servers return JSON with incorrect content-type headers
                try:
                    result = await response.json()
                except aiohttp.ContentTypeError:
                    # If JSON parsing fails due to content-type, try parsing text as JSON
                    try:
                        response_text = await response.text()
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        content_type = response.headers.get("content-type", "")
                        raise MCPConnectionError(
                            f"Server returned non-JSON response (content-type: {content_type}). "
                            f"Response body: {response_text[:500]}..."
                        )

                if "error" in result:
                    raise MCPError(
                        result["error"]["code"],
                        result["error"]["message"],
                        result["error"].get("data"),
                    )

                return result.get("result", {})

        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")

    async def initialize(self) -> Dict:
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "MCP HTTP Client", "version": "1.0.0"},
            },
        )
        self._initialized = True
        return result

    async def list_tools(self) -> List[MCPTool]:
        if not self._initialized:
            await self.initialize()

        result = await self._send_request("tools/list")
        return [
            MCPTool(
                name=tool["name"],
                description=tool["description"],
                input_schema=tool["inputSchema"],
            )
            for tool in result.get("tools", [])
        ]

    def export_tools_for_llm(
        self, llm_name: str, tools: List[MCPTool]
    ) -> List[Dict[str, Any]]:
        tools_for_llm = []
        if llm_name == "gemini":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                )
        elif llm_name == "claude":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
        elif llm_name == "gpt":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                )
        elif llm_name == "ollama":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "type": "function",
                        "function": {
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            }
                        },
                    }
                )
        else:
            raise MCPError("LLM not supported")
        return tools_for_llm

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        if not self._initialized:
            await self.initialize()

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        result = await self._send_request("tools/call", params)
        return result.get("content", [])

    async def list_resources(self) -> List[MCPResource]:
        if not self._initialized:
            await self.initialize()

        result = await self._send_request("resources/list")
        return [
            MCPResource(
                uri=res["uri"],
                name=res["name"],
                description=res["description"],
                mime_type=res["mimeType"],
            )
            for res in result.get("resources", [])
        ]

    async def read_resource(self, uri: str) -> Dict:
        if not self._initialized:
            await self.initialize()

        return await self._send_request("resources/read", {"uri": uri})

    async def list_prompts(self) -> List[MCPPrompt]:
        if not self._initialized:
            await self.initialize()

        result = await self._send_request("prompts/list")
        return [
            MCPPrompt(
                name=prompt["name"],
                description=prompt["description"],
                arguments=prompt.get("arguments", []),
            )
            for prompt in result.get("prompts", [])
        ]

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> Dict:
        if not self._initialized:
            await self.initialize()

        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        return await self._send_request("prompts/get", params)

    async def health_check(self) -> Dict:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"Health check failed: {e}")

    async def get_server_info(self) -> Dict:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        try:
            async with self.session.get(f"{self.base_url}/") as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise MCPConnectionError(f"Failed to get server info: {e}")


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class MCPConnectionError(Exception):
    pass
