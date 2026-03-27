"""
Implement Node
Reads: current_plan, code_base, restructured_query, implementation_status
Writes: files_modified, implement_iteration_count, messages, execution_log
Uses: tool_calling_node (read, write, search, shell commands, etc.)
"""

import logging
import copy
import queue
import threading

from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm
from mcp_client.client import MCPClient
from tools.tool_calling_node import build_implement_prompts, run_tool_calling_loop
from core.async_utils import run_async

logger = logging.getLogger(__name__)

# Thread-safe queue for confirm-mode: background thread posts requests,
# main thread (via CLI) responds. Shared module-level so CLI can access it.
_confirm_request_q: queue.Queue = queue.Queue()
_confirm_response_q: queue.Queue = queue.Queue()


def _confirm_callback(tool_name: str, args_preview: str) -> bool:
    """
    Called from background async thread during confirm mode.
    Posts a confirmation request and blocks until main thread responds.
    """
    _confirm_request_q.put((tool_name, args_preview))
    return _confirm_response_q.get()  # blocks until CLI answers


def _stream_callback(text: str) -> None:
    """Stream final LLM summary text to stdout."""
    import sys
    sys.stdout.write(f"   {text}\n")
    sys.stdout.flush()


def run(state: AgentState) -> dict:
    """
    Execute the plan by running an LLM tool-calling loop over MCP tools.
    """
    config = load_config()
    route_type = state.get("route_type", "")
    # Simple → Ollama, Complex → cloud (OpenAI/Groq)
    if route_type == "complex":
        llm = get_llm(config, route_type=route_type, force_cloud=True)
        fallback_llm = get_llm(config, route_type="simple")  # Ollama fallback for rate limits
    else:
        llm = get_llm(config, route_type="simple")  # Ollama
        fallback_llm = None
    error_log = list(state.get("error_log", []) or [])

    current_plan = state.get("current_plan", "")
    restructured_query = state.get("restructured_query", state.get("user_query", ""))
    code_base = state.get("code_base", {})
    # Read execution_mode: prefer state (set by CLI /mode command), fallback to config
    execution_mode = state.get("execution_mode") or config.get("execution_mode", "auto")
    print(f"   [execution_mode: {execution_mode}]", flush=True)

    implement_iteration = int(state.get("implement_iteration_count", 0)) + 1

    # Override MCP filesystem server path to point at user's project
    runtime_config = copy.deepcopy(config)
    project_path = runtime_config.get("project_path", ".")
    fs_server = (runtime_config.get("mcp_servers") or {}).get("filesystem")
    if fs_server and fs_server.get("args"):
        fs_server["args"] = [
            project_path if arg == "." else arg for arg in fs_server["args"]
        ]

    rag_context = state.get("rag_query_results", []) or []

    # Set up confirm callback only in confirm mode
    confirm = (execution_mode or "").lower() in {"confirm", "confirmation"}
    confirm_fn = _confirm_callback if confirm else None

    # Run MCP connect, tool loop, and cleanup in a single async task
    # to avoid anyio cancel scope issues across tasks
    async def _run_with_mcp():
        mcp_client = MCPClient(runtime_config)
        try:
            await mcp_client.connect_all()
            available_tools = mcp_client.get_tools()
            server_status = mcp_client.get_server_status()
            print(f"   MCP connected. Tools: {len(available_tools)}. Calling LLM...", flush=True)

            system_prompt, user_prompt = build_implement_prompts(
                restructured_query=restructured_query,
                current_plan=current_plan,
                code_base=code_base,
                available_mcp_tools=available_tools,
                project_path=project_path,
                rag_context=rag_context,
            )

            execution_log, files_modified, messages = await run_tool_calling_loop(
                llm=llm,
                mcp_client=mcp_client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                execution_mode=execution_mode,
                max_steps=25,
                fallback_llm=fallback_llm,
                confirm_fn=confirm_fn,
                on_stream_token=_stream_callback,
            )

            return {
                "execution_log": execution_log,
                "files_modified": files_modified,
                "messages": messages,
                "available_tools": available_tools,
                "server_status": server_status,
            }
        finally:
            await mcp_client.cleanup()

    result = run_async(_run_with_mcp())

    logger.info("Implement iteration %d complete. Tool calls: %d",
                implement_iteration, len(result["execution_log"]))

    return {
        "files_modified": result["files_modified"],
        "implement_iteration_count": implement_iteration,
        "messages": result["messages"],
        "execution_log": result["execution_log"],
        "available_mcp_tools": result["available_tools"],
        "mcp_server_status": result["server_status"],
        "current_node": "implement",
        "error_log": error_log,
    }
