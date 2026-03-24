"""
Code Judge Node
===============
Reads:  files_modified, current_plan, restructured_query, execution_log
Writes: implementation_status
"""

import logging
from datetime import datetime

import anthropic

from core.state import AgentState

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a senior code reviewer and QA engineer.
You are given a coding task, the plan that was followed, and the files that were modified.
You have access to tools to read files and run commands to verify the implementation.

Your job:
1. Read every modified file to inspect the changes.
2. Run relevant commands to verify correctness (e.g. python -m pytest, python file.py, etc).
3. Check that the implementation actually satisfies the original task and plan.
4. Return a JSON verdict in this exact format (no extra text):

{
  "status": "correct" | "incorrect",
  "feedback": "Brief explanation of what is right or wrong.",
  "issues": ["issue 1", "issue 2"]  // empty list if correct
}
"""


def _build_prompt(state: AgentState) -> str:
    query         = state.get("restructured_query") or state.get("user_query", "")
    plan          = state.get("current_plan", "No plan provided.")
    files         = state.get("files_modified") or []
    execution_log = state.get("execution_log") or []
    iteration     = state.get("implement_iteration_count", 1)

    log_summary = ""
    if execution_log:
        last_entries = execution_log[-10:]  # only last 10 to keep prompt tight
        lines = [
            f"- [{e['tool']}] {str(e['input'])[:80]} → {str(e['result'])[:120]}"
            for e in last_entries
        ]
        log_summary = "\n## Execution Log (last 10 steps)\n" + "\n".join(lines)

    files_list = "\n".join(f"- {f}" for f in files) or "None"

    return (
        f"## Original Task\n{query}\n\n"
        f"## Implementation Plan\n{plan}\n\n"
        f"## Files Modified (iteration {iteration})\n{files_list}"
        f"{log_summary}\n\n"
        "Review the implementation now. Read the files, run verification commands, "
        "then return your JSON verdict."
    )


async def run(state: AgentState) -> dict:
    """
    Code judge node — reads modified files, runs verification commands,
    and uses Claude to produce a pass/fail verdict with feedback.
    """
    mcp_client = state.get("_mcp_client")
    tools      = state.get("available_mcp_tools") or []
    client     = anthropic.Anthropic()

    messages = [{"role": "user", "content": _build_prompt(state)}]

    final_text = ""

    # ── Agentic tool-use loop ─────────────────────────────────────────────────
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

                logger.info("code_judge tool call: %s | %s", block.name, block.input)

                try:
                    if mcp_client:
                        result = await mcp_client.call_tool(block.name, block.input)
                    else:
                        result = f"[error] MCP client not available"
                except Exception as exc:
                    result = f"[error] {exc}"
                    logger.exception("Tool %s failed in code_judge", block.name)

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            break

    # ── Parse verdict ─────────────────────────────────────────────────────────
    import json, re

    verdict = _parse_verdict(final_text)

    implementation_status = {
        "status":         verdict.get("status", "incorrect"),
        "feedback":       verdict.get("feedback", final_text),
        "issues":         verdict.get("issues", []),
        "reviewed_at":    datetime.utcnow().isoformat(),
        "files_reviewed": state.get("files_modified") or [],
        "iteration":      state.get("implement_iteration_count", 1),
    }

    logger.info("code_judge verdict: %s", implementation_status["status"])

    return {
        "implementation_status": implementation_status,
        "current_node": "code_judge",
    }


def _parse_verdict(text: str) -> dict:
    """Extract JSON verdict from Claude's response, with fallback."""
    import json, re

    # Try to find a JSON block in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback — treat as incorrect with raw feedback
    logger.warning("code_judge could not parse JSON verdict, marking as incorrect")
    return {
        "status":   "incorrect",
        "feedback": text.strip() or "No feedback returned.",
        "issues":   ["Could not parse verdict from LLM response."],
    }