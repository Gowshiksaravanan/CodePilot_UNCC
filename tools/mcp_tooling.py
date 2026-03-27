"""
MCP Tooling Helpers
Build LangChain-compatible tool(s) that route into MCPClient.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool


def _mcp_result_to_text(result: Any) -> str:
    """
    Convert an MCP CallToolResult to a text blob for ToolMessage.
    """
    blocks = getattr(result, "content", None) or []
    texts: list[str] = []
    for block in blocks:
        if hasattr(block, "text"):
            texts.append(block.text)
    if texts:
        return "\n".join(texts).strip()
    return str(result)


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def build_mcp_call_tool(mcp_client) -> Any:
    """
    Return a single generic tool `mcp_call(tool_name, arguments)` that the LLM can use
    to call any discovered MCP tool.
    """

    @tool("mcp_call")
    def mcp_call(tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call an MCP tool by name.

        Args:
          tool_name: MCP tool name (e.g. 'read_file', 'edit_file', 'tavily_search', 'query_python_docs')
          arguments: JSON object of tool arguments (must match the MCP tool schema)
        Returns:
          Tool output as text (usually JSON).
        """
        from core.async_utils import run_async

        result = run_async(mcp_client.call_tool(tool_name, arguments))
        return _mcp_result_to_text(result)

    return mcp_call


def build_tools_summary(available_mcp_tools: List[dict]) -> str:
    """
    Produce a compact tool list for prompts.
    Only includes tool name, short description, and required parameters
    to keep the prompt small for local models.
    """
    lines: list[str] = []
    for t in available_mcp_tools:
        name = t.get("name", "")
        desc = t.get("description", "")[:120]
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        params = ", ".join(f"{k}: {props[k].get('type','any')}" for k in required if k in props)
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)

