"""
MCP Client
Connects to all MCP servers defined in config.yaml at startup.
Discovers tools dynamically and provides a unified tool list to the agent.
"""


class MCPClient:
    def __init__(self, config: dict):
        """TODO: Initialize connections to all configured MCP servers."""
        self.config = config
        self.tools = []

    def connect(self):
        """TODO: Connect to each MCP server and discover available tools."""
        pass

    def get_tools(self) -> list:
        """TODO: Return unified list of all tools from all servers."""
        return self.tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """TODO: Route tool call to the correct MCP server and return result."""
        return ""
