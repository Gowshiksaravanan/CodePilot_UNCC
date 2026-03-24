"""
Implement Node
==============
Reads:  current_plan, code_base, restructured_query, implementation_status
Writes: files_modified, implement_iteration_count, messages, execution_log
Uses:   MCPClient (read, write, search, shell commands, etc.)
"""

import json
import logging
from datetime import datetime

import anthropic

from core.state import AgentState

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are CodePilot, an expert coding agent.
You are given a task, an implementation plan, and access to tools.

Your job is to execute the plan exactly — step by step — using the available tools.
- Use read_file to inspect files before editing them.
- Use write_file to apply changes.
- Use run_command to install dependencies, run tests, or verify output.
- After completing all steps, reply with a concise summary of every file you changed.

Be precise. Do not skip steps. Do not modify files not mentioned in the plan.
"""


def _build_prompt(state: AgentState) -> str:
    query  = state.get("restructured_query") or state.get("user_query", "")
    plan   = state.get("current_plan", "No plan provided.")
    status = state.get("implementation_status") or {}
    count  = state.get("implement_iteration_count", 0)

    code_base = state.get("code_base") or {}
    code_section = ""
    if code_base:
        snippets = "\n\n".join(
            f"### {path}\n```\n{content}\n```"
            for path, content in code_base.items()
        )
        code_section = f"\n\n## Existing Code\n{snippets}"

    retry_section = ""
    if count > 0 and status:
        feedback = status.get("feedback", "No feedback provided.")
        retry_section = f"\n\n## Previous Attempt Feedback\n{feedback}\nPlease fix the issues described above."

    return (
        f"## Task\n{query}\n\n"
        f"## Implementation Plan\n{plan}"
        f"{code_section}"
        f"{retry_section}\n\n"
        "Execute the plan now using the available tools."
    )


async def run(state: AgentState) -> dict:
    """
    Implementation node — executes the plan via agentic tool-use loop.
    Returns state updates for: files_modified, implement_iteration_count,
                               messages, execution_log, implementation_status.
    """
    mcp_client = state.get("_mcp_client")  # injected at graph startup
    tools      = state.get("available_mcp_tools") or []
    client     = anthropic.Anthropic()

    messages: list       = list(state.get("messages") or [])
    execution_log: list  = list(state.get("execution_log") or [])
    files_modified: list = list(state.get("files_modified") or [])
    iteration_count: int = state.get("implement_iteration_count", 0) + 1

    prompt = _build_prompt(state)
    messages.append({"role": "user", "content": prompt})

    # ── Agentic tool-use loop ─────────────────────────────────────────────────
    final_text = ""

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text
                
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                called_at  = datetime.utcnow().isoformat()

                logger.info("Tool call: %s | input: %s", tool_name, tool_input)

                try:
                    if mcp_client:
                        result = await mcp_client.call_tool(tool_name, tool_input)
                    else:
                        result = f"[error] MCP client not available — cannot call {tool_name}"
                except Exception as exc:
                    result = f"[error] {exc}"
                    logger.exception("Tool %s failed", tool_name)

                # Track write_file calls -> files_modified
                if tool_name == "write_file":
                    path = tool_input.get("path", "")
                    if path and path not in files_modified:
                        files_modified.append(path)

                execution_log.append({
                    "tool":      tool_name,
                    "input":     tool_input,
                    "result":    result[:500],
                    "timestamp": called_at,
                })

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            messages.append({"role": "assistant", "content": final_text})
            break

    # ── Build implementation_status for code_judge ────────────────────────────
    implementation_status = {
        "status":         "pending_review",
        "summary":        final_text,
        "files_modified": files_modified,
        "iteration":      iteration_count,
    }

    return {
        "messages":                  messages,
        "files_modified":            files_modified,
        "implement_iteration_count": iteration_count,
        "execution_log":             execution_log,
        "implementation_status":     implementation_status,
        "current_node":              "implement",
    }