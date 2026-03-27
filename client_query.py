import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        # Run the server file directly so imports work without setting PYTHONPATH.
        args=["CodePilot_UNCC/mcp_servers/rag_server/server.py"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.call_tool(
                "query_python_docs",
                arguments={"query": "How do decorators work in Python?"},
            )

            text = "".join(block.text for block in response.content if hasattr(block, "text"))
            print("Raw response:")
            print(text)

            try:
                payload = json.loads(text)
                print("\nParsed payload:")
                print(json.dumps(payload, indent=2))
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())

