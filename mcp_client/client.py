"""
MCP Client
Connects to all MCP servers defined in config.yaml at startup.
Discovers tools dynamically and provides a unified tool list to the agent.
"""

import logging
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, config: dict):
        self.config = config
        self.sessions: dict[str, ClientSession] = {}
        self.tools_map: dict[str, str] = {}  # tool_name -> server_name
        self._tools: list[dict] = []
        self._exit_stack = AsyncExitStack()

    async def connect_all(self):
        """Connect to each MCP server defined in config and discover tools."""
        servers = self.config.get("mcp_servers", {})
        for name, server_config in servers.items():
            try:
                await self._connect_server(name, server_config)
            except Exception:
                logger.exception("Failed to connect to MCP server '%s'", name)

    async def _connect_server(self, name: str, server_config: dict):
        """Connect to a single MCP server via stdio transport."""
        env = {**os.environ, **(server_config.get("env") or {})}
        params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=env,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self.sessions[name] = session

        result = await session.list_tools()
        for tool in result.tools:
            self.tools_map[tool.name] = name
            self._tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            })

        logger.info(
            "Connected to MCP server '%s' — discovered %d tools",
            name,
            len(result.tools),
        )

    def get_tools(self) -> list[dict]:
        """Return unified list of all tools from all connected servers."""
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP server and return the result."""
        server_name = self.tools_map.get(tool_name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments=arguments)

        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )

    async def cleanup(self):
        """Close all server connections."""
        await self._exit_stack.aclose()
