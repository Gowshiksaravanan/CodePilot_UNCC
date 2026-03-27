"""
MCP Client
Connects to all MCP servers defined in config.yaml at startup.
Discovers tools dynamically and provides a unified tool list to the agent.
Handles reconnection on failure per design doc.
"""

import asyncio
import logging
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


class MCPClient:
    def __init__(self, config: dict):
        self.config = config
        self.sessions: dict[str, ClientSession] = {}
        self.tools_map: dict[str, str] = {}  # tool_name -> server_name
        self._tools: list[dict] = []
        self._exit_stack = AsyncExitStack()
        self._server_configs: dict[str, dict] = {}  # name -> server_config for reconnection

    async def connect_all(self, only_servers: list[str] | None = None):
        """Connect to MCP servers (optionally filtered) and discover tools."""
        servers = self.config.get("mcp_servers", {})
        for name, server_config in servers.items():
            if only_servers is not None and name not in set(only_servers):
                continue
            self._server_configs[name] = server_config
            await self._connect_with_retry(name, server_config)

    async def _connect_with_retry(self, name: str, server_config: dict):
        """Attempt to connect to a server with retries."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await self._connect_server(name, server_config)
                return
            except Exception:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Failed to connect to MCP server '%s' (attempt %d/%d). Retrying in %ds...",
                        name, attempt, MAX_RETRIES, RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.exception(
                        "Failed to connect to MCP server '%s' after %d attempts.", name, MAX_RETRIES,
                    )

    async def _connect_server(self, name: str, server_config: dict):
        """Connect to a single MCP server via stdio transport."""
        env = {**os.environ, **(server_config.get("env") or {})}
        params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=env,
        )

        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self.sessions[name] = session

        result = await session.list_tools()
        for tool in result.tools:
            self.tools_map[tool.name] = name
            self._tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )

        logger.info(
            "Connected to MCP server '%s' — discovered %d tools",
            name,
            len(result.tools),
        )

    def get_tools(self) -> list[dict]:
        """Return unified list of all tools from all connected servers."""
        return self._tools

    def get_server_status(self) -> dict[str, str]:
        """Return connection status of each configured server."""
        all_servers = self.config.get("mcp_servers", {})
        return {
            name: ("connected" if name in self.sessions else "disconnected")
            for name in all_servers
        }

    async def call_tool(self, tool_name: str, arguments: dict):
        """Route a tool call to the correct MCP server. Attempts reconnection on failure."""
        server_name = self.tools_map.get(tool_name)
        if server_name is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        session = self.sessions.get(server_name)
        if session is None:
            # Server was disconnected — attempt reconnection
            server_config = self._server_configs.get(server_name)
            if server_config:
                logger.warning("Server '%s' disconnected. Attempting reconnection...", server_name)
                await self._connect_with_retry(server_name, server_config)
                session = self.sessions.get(server_name)

            if session is None:
                raise ConnectionError(
                    f"MCP server '{server_name}' is disconnected and reconnection failed."
                )

        try:
            return await session.call_tool(tool_name, arguments=arguments)
        except Exception as exc:
            # Attempt single reconnect on call failure
            logger.warning(
                "Tool call '%s' failed on server '%s': %s. Attempting reconnect...",
                tool_name, server_name, exc,
            )
            server_config = self._server_configs.get(server_name)
            if server_config:
                # Remove stale session
                self.sessions.pop(server_name, None)
                try:
                    await self._connect_server(server_name, server_config)
                    session = self.sessions.get(server_name)
                    if session:
                        return await session.call_tool(tool_name, arguments=arguments)
                except Exception:
                    logger.exception("Reconnection to '%s' failed.", server_name)
            raise

    async def cleanup(self):
        """Close all server connections."""
        await self._exit_stack.aclose()
