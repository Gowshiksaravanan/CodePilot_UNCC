"""
Custom RAG MCP Server
Exposes exactly one tool:
query_python_docs(query: str)
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

# Allow running as a script: ensure CodePilot_UNCC/ is on sys.path
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from mcp_servers.rag_server.retriever import retrieve


mcp = FastMCP("python-docs-rag")


@mcp.tool()
def query_python_docs(query: str) -> dict:
    """
    Retrieve Python documentation chunks for a free-text query.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "error": "query must be a non-empty string",
            "results": [],
        }

    try:
        return retrieve(query=query.strip(), top_k=5)
    except FileNotFoundError as exc:
        return {
            "error": str(exc),
            "results": [],
        }
    except Exception as exc:
        return {
            "error": f"retrieval failed: {exc}",
            "results": [],
        }


def main() -> None:
    """
    Start the MCP server over stdio transport.
    """
    mcp.run()


if __name__ == "__main__":
    main()
