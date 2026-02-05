#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from silvaengine_utility import Debugger, Serializer, convert_decimal_to_number

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
        # Create session optimized for AWS Lambda
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

    def _to_snake_case(self, camel_str: str) -> str:
        """Convert camelCase string to snake_case."""
        import re

        # Insert underscore before uppercase letters and convert to lowercase
        snake_str = re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()
        return snake_str

    def _normalize_schema_keywords(self, schema: Any) -> Any:
        """
        Recursively convert keyword values to proper data types in JSON Schema.
        Handles nested arrays and objects.
        Converts property keys from camelCase to snake_case.
        """
        if not isinstance(schema, dict):
            return schema

        normalized = {}

        # Keywords that should be integers
        integer_keywords = {
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "minProperties",
            "maxProperties",
            "minContains",
            "maxContains",
        }

        # Keywords that should be numbers (int or float)
        number_keywords = {
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
        }

        # Keywords that should be booleans
        boolean_keywords = {"uniqueItems", "additionalProperties"}

        # Keywords that should be decimals/numbers
        decimal_keywords = {"default", "const"}

        for key, value in schema.items():
            # Convert to proper data type based on keyword
            if key in integer_keywords:
                if isinstance(value, int):
                    normalized[key] = value
                else:
                    try:
                        normalized[key] = int(value)
                    except (ValueError, TypeError):
                        normalized[key] = value
            elif key in number_keywords:
                if isinstance(value, (int, float)):
                    normalized[key] = value
                else:
                    try:
                        # Try int first, then float
                        normalized[key] = (
                            int(value)
                            if isinstance(value, str) and value.isdigit()
                            else float(value)
                        )
                    except (ValueError, TypeError):
                        normalized[key] = value
            elif key in decimal_keywords:
                # Handle default and const values - convert if they are numeric strings
                if isinstance(value, (int, float)):
                    normalized[key] = value
                elif isinstance(value, str):
                    try:
                        # Try to convert to number if it's a numeric string
                        normalized[key] = (
                            int(value) if value.isdigit() else float(value)
                        )
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        normalized[key] = value
                else:
                    # Keep the value as-is for other types (bool, list, dict, etc.)
                    normalized[key] = value
            elif key in boolean_keywords:
                if isinstance(value, str):
                    normalized[key] = value.lower() in ("true", "1", "yes")
                elif isinstance(value, bool):
                    normalized[key] = value
                else:
                    try:
                        normalized[key] = bool(value)
                    except (ValueError, TypeError):
                        normalized[key] = value
            # Recursive handling for nested structures
            elif key == "properties" and isinstance(value, dict):
                # Recursively normalize each property schema and convert keys to snake_case
                normalized[key] = {
                    self._to_snake_case(prop_name): self._normalize_schema_keywords(
                        prop_schema
                    )
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                # Recursively normalize array item schema
                normalized[key] = self._normalize_schema_keywords(value)
            elif key == "additionalItems" and isinstance(value, dict):
                # Recursively normalize additional items schema
                normalized[key] = self._normalize_schema_keywords(value)
            elif key == "patternProperties" and isinstance(value, dict):
                # Recursively normalize pattern properties
                normalized[key] = {
                    pattern: self._normalize_schema_keywords(prop_schema)
                    for pattern, prop_schema in value.items()
                }
            elif key == "contains" and isinstance(value, dict):
                # Recursively normalize contains schema
                normalized[key] = self._normalize_schema_keywords(value)
            elif key == "allOf" and isinstance(value, list):
                # Recursively normalize allOf schemas
                normalized[key] = [
                    self._normalize_schema_keywords(item) for item in value
                ]
            elif key == "anyOf" and isinstance(value, list):
                # Recursively normalize anyOf schemas
                normalized[key] = [
                    self._normalize_schema_keywords(item) for item in value
                ]
            elif key == "oneOf" and isinstance(value, list):
                # Recursively normalize oneOf schemas
                normalized[key] = [
                    self._normalize_schema_keywords(item) for item in value
                ]
            elif key == "not" and isinstance(value, dict):
                # Recursively normalize not schema
                normalized[key] = self._normalize_schema_keywords(value)
            else:
                # Keep other values as-is
                normalized[key] = value

        return normalized

    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
        }
        if params:
            request_data["params"] = self._clean_params(params)
            # if cleaned_params:
            #     request_data["params"] = convert_decimal_to_number(cleaned_params)

        headers = {
            "Content-Type": "application/json",
            **self.custom_headers,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        try:
            async with self.session.post(
                self.base_url,
                json=Serializer.json_dumps(request_data),
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
                input_schema=tool.get("input_schema", tool.get("inputSchema", {})),
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
                        "parameters": self._normalize_schema_keywords(
                            tool.input_schema
                        ),
                    }
                )
        elif llm_name == "claude":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": self._normalize_schema_keywords(
                            tool.input_schema
                        ),
                    }
                )
        elif llm_name == "gpt":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._normalize_schema_keywords(
                            tool.input_schema
                        ),
                    }
                )
        elif llm_name == "ollama":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": self._normalize_schema_keywords(
                                tool.input_schema
                            ),
                        },
                    }
                )
        elif llm_name == "travrse":
            for tool in tools:
                tools_for_llm.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "tool_type": "local",
                        "parameters_schema": self._normalize_schema_keywords(
                            tool.input_schema
                        ),
                    }
                )
        else:
            raise MCPError("LLM not supported")
        return tools_for_llm

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        try:
            if not self._initialized:
                await self.initialize()

            params = {"name": name}

            if arguments:
                params["arguments"] = arguments

            result = await self._send_request("tools/call", params)
            return result.get("content", [])
        except Exception as e:
            Debugger.info(
                variable=f"Error: {str(e)}, Params: {params}, Details: {e}",
                stage=f"{__file__}.call_tool",
            )

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
