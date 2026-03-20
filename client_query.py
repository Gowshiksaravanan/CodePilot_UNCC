import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.rag_server.server"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.call_tool(
                "query_python_docs",
                arguments={"query": "How do decorators work in Python?"},
            )

            # MCP tool responses are typically text blocks
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            print("Raw response:")
            print(text)

            # Optional: pretty-print JSON if returned
            try:
                payload = json.loads(text)
                print("\nParsed payload:")
                print(json.dumps(payload, indent=2))
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())
