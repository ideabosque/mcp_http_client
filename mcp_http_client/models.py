from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str


@dataclass
class MCPPrompt:
    name: str
    description: str
    arguments: List[Dict[str, Any]]
