"""
Tool-calling execution helper for implement node.
Runs an LLM loop with a generic `mcp_call` tool.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools.mcp_tooling import build_mcp_call_tool, build_tools_summary


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_tool_call_from_text(text: str) -> List[Dict[str, Any]]:
    """Fallback: extract tool calls from text when the model outputs JSON instead of using tool calling API."""
    import re
    import uuid

    calls = []

    # Pattern 1: Groq-style <function=name{...}</function>
    func_pattern = re.findall(r'<function=(\w+)(.*?)</function>', text, re.DOTALL)
    for func_name, func_args_str in func_pattern:
        try:
            data = json.loads(func_args_str)
            if func_name == "mcp_call" and "tool_name" in data:
                calls.append({
                    "name": "mcp_call",
                    "args": data,
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                })
            else:
                calls.append({
                    "name": "mcp_call",
                    "args": {"tool_name": func_name, "arguments": data},
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                })
        except json.JSONDecodeError:
            continue
    if calls:
        return calls

    # Pattern 2: JSON objects with name/tool_name fields
    def find_json_objects(s: str) -> List[str]:
        results = []
        i = 0
        while i < len(s):
            if s[i] == '{':
                depth = 0
                start = i
                while i < len(s):
                    if s[i] == '{':
                        depth += 1
                    elif s[i] == '}':
                        depth -= 1
                    if depth == 0:
                        results.append(s[start:i+1])
                        break
                    i += 1
            i += 1
        return results

    for candidate in find_json_objects(text):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue
            name = data.get("name") or data.get("tool_name")
            if not name:
                continue
            args = data.get("parameters") or data.get("arguments") or data.get("args") or {}
            calls.append({
                "name": "mcp_call",
                "args": {"tool_name": name, "arguments": args},
                "id": str(uuid.uuid4()),
                "type": "tool_call",
            })
        except json.JSONDecodeError:
            continue

    return calls


def _extract_files_modified(tool_name: str, arguments: Dict[str, Any]) -> List[str]:
    path_keys = ["path", "file_path", "source", "destination", "old_path", "new_path"]
    modified = []
    if tool_name in {"write_file", "edit_file", "move_file", "create_directory"}:
        for k in path_keys:
            v = arguments.get(k)
            if isinstance(v, str) and v:
                modified.append(v)
    return modified


def _mcp_result_to_text(result: Any) -> str:
    """Convert an MCP CallToolResult to text."""
    blocks = getattr(result, "content", None) or []
    texts: list[str] = []
    for block in blocks:
        if hasattr(block, "text"):
            texts.append(block.text)
    if texts:
        return "\n".join(texts).strip()
    return str(result)


def _request_confirmation_from_main_thread(
    tool_name: str, args_preview: str, confirm_fn: Optional[Callable]
) -> bool:
    """
    Ask user for tool confirmation. Uses confirm_fn callback if provided
    (runs on main thread via queue), otherwise auto-approves.
    """
    if confirm_fn is None:
        return True

    # Use a thread-safe queue to get the answer from the main thread
    result_q: queue.Queue = queue.Queue()

    def _ask():
        try:
            answer = confirm_fn(tool_name, args_preview)
            result_q.put(answer)
        except Exception:
            result_q.put(True)  # auto-approve on error

    # Run on main thread if we're in a background thread
    if threading.current_thread() is not threading.main_thread():
        # Can't call input() from background thread, so use confirm_fn
        # which should be thread-safe (e.g., using queue internally)
        return confirm_fn(tool_name, args_preview)
    else:
        _ask()
        return result_q.get()


async def run_tool_calling_loop(
    *,
    llm,
    mcp_client,
    system_prompt: str,
    user_prompt: str,
    execution_mode: str,
    max_steps: int = 25,
    fallback_llm=None,
    confirm_fn: Optional[Callable] = None,
    on_stream_token: Optional[Callable] = None,
) -> Tuple[List[dict], List[str], List]:
    """
    Async tool calling loop. Calls MCP tools directly via await
    to avoid deadlocking the event loop.

    Args:
        confirm_fn: Callback (tool_name, args_preview) -> bool for confirm mode.
                    Called from background thread; must be thread-safe.
        on_stream_token: Callback (token: str) -> None for streaming final response.

    Returns: (execution_log, files_modified, messages)
    """
    tool = build_mcp_call_tool(mcp_client)
    bound = llm.bind_tools([tool])

    confirm = (execution_mode or "").lower() in {"confirm", "confirmation"}

    messages: List = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    execution_log: List[dict] = []
    files_modified: List[str] = []
    consecutive_errors = 0
    last_error_msg = ""

    for step_i in range(max_steps):
        # Break out of repeated-failure loops
        if consecutive_errors >= 3:
            correction = (
                f"STOP: The last {consecutive_errors} tool calls all failed with the same error. "
                f"Error: {last_error_msg[:300]}\n"
                "You MUST fix the argument format. For mcp_call, use: "
                '{"tool_name": "<name>", "arguments": {"param": "value"}}. '
                "For create_directory, arguments must be: {\"path\": \"/absolute/path\"}. "
                "For write_file, arguments must be: {\"path\": \"/absolute/path\", \"content\": \"...\"}. "
                "If you cannot fix the issue, respond with a text summary instead of more tool calls."
            )
            messages.append(HumanMessage(content=correction))
            consecutive_errors = 0
            print(f"   [injected error correction hint]", flush=True)
        print(f"   LLM call #{step_i + 1}...", flush=True)
        ai = None
        tool_calls = []
        try:
            ai = bound.invoke(messages)
        except Exception as invoke_err:
            err_str = str(invoke_err)

            # Rate limit: fall back to Ollama if available
            if ("rate_limit" in err_str.lower() or "429" in err_str) and fallback_llm is not None:
                print("   Cloud API rate limited — falling back to Ollama...", flush=True)
                bound = fallback_llm.bind_tools([tool])
                try:
                    ai = bound.invoke(messages)
                except Exception:
                    raise invoke_err

            # Groq sometimes returns tool_use_failed with the generation in the error body.
            else:
                err_body = getattr(invoke_err, "body", None) or {}
                failed_gen = ""
                if isinstance(err_body, dict):
                    failed_gen = (err_body.get("error") or {}).get("failed_generation", "")
                if not failed_gen:
                    failed_gen = err_str
                parsed = _parse_tool_call_from_text(failed_gen)
                if parsed:
                    ai = AIMessage(content=failed_gen, tool_calls=[])
                    messages.append(ai)
                    tool_calls = parsed
                else:
                    raise

        if ai is not None and not tool_calls:
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls and ai.content:
                tool_calls = _parse_tool_call_from_text(ai.content)

        if not tool_calls:
            # Stream the final summary response token-by-token if callback provided
            if ai and ai.content and on_stream_token:
                on_stream_token(ai.content)
            elif ai and ai.content:
                print(f"   {ai.content[:300]}", flush=True)
            else:
                print(f"   No more tool calls. Done.", flush=True)
            break

        print(f"   Got {len(tool_calls)} tool call(s)", flush=True)
        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or str(id(call))

            # Resolve actual MCP tool name and arguments
            if tool_name == "mcp_call" and "tool_name" in args:
                actual_tool = args["tool_name"]
                actual_args = args.get("arguments", {})
                # Fix: if arguments is empty but extra keys exist beside tool_name,
                # the model likely flattened args (e.g. {"tool_name": "create_directory", "path": "/x"})
                if not actual_args:
                    extra = {k: v for k, v in args.items() if k not in ("tool_name", "arguments")}
                    if extra:
                        actual_args = extra
            else:
                actual_tool = tool_name
                actual_args = args

            args_preview = json.dumps(actual_args, default=str)[:200]
            print(f"   → {actual_tool}({args_preview})", flush=True)

            # Confirm mode: ask user before executing
            if confirm and confirm_fn:
                approved = confirm_fn(actual_tool, args_preview)
                if not approved:
                    tool_output = "User rejected tool call."
                    print(f"     [rejected]", flush=True)
                    messages.append(ToolMessage(content=tool_output, tool_call_id=call_id))
                    execution_log.append({
                        "timestamp": _now_iso(),
                        "tool": actual_tool,
                        "input": actual_args,
                        "output": tool_output,
                        "status": "rejected",
                    })
                    continue

            try:
                # Call MCP directly via await — no run_async needed, avoids deadlock
                result = await mcp_client.call_tool(actual_tool, actual_args)
                tool_output = _mcp_result_to_text(result)
                status = "success"
            except Exception as exc:
                tool_output = f"ERROR: {exc}"
                status = "error"

            print(f"     {status}: {str(tool_output)[:150]}", flush=True)
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))
            execution_log.append(
                {
                    "timestamp": _now_iso(),
                    "tool": actual_tool,
                    "input": actual_args,
                    "output": tool_output,
                    "status": status,
                }
            )
            files_modified.extend(_extract_files_modified(actual_tool, actual_args))

            # Track consecutive errors for loop-breaking
            if status == "error" or "error" in str(tool_output).lower()[:100]:
                consecutive_errors += 1
                last_error_msg = str(tool_output)[:500]
            else:
                consecutive_errors = 0

    # de-dup while keeping order
    seen = set()
    files_modified = [p for p in files_modified if not (p in seen or seen.add(p))]

    return execution_log, files_modified, messages


def build_implement_prompts(
    *,
    restructured_query: str,
    current_plan: str,
    code_base: dict,
    available_mcp_tools: List[dict],
    project_path: str = ".",
    rag_context: List[dict] | None = None,
) -> Tuple[str, str]:
    tools_summary = build_tools_summary(available_mcp_tools)

    system_prompt = (
        "You are CodePilot Implementer.\n"
        "Execute the approved plan by calling tools. Use ONLY the tool `mcp_call`.\n"
        "Do not pretend to have edited files—always call tools.\n"
        "Use filesystem MCP tools (write_file, edit_file, read_file) for code changes.\n"
        f"IMPORTANT: All file paths MUST be absolute, under the project directory: {project_path}\n"
        f"Example: write_file with path '{project_path}/filename.py'\n"
        "Only call query_python_docs or tavily_search if you genuinely need API reference.\n"
        "For simple tasks, just write the code directly with write_file.\n"
        "After completing steps, respond with a short completion summary.\n"
    )

    # Build RAG context section if available
    rag_section = ""
    if rag_context:
        chunks = []
        for r in rag_context[:5]:
            chunk = r.get("chunk", r.get("snippet", ""))
            source = r.get("source", "")
            if chunk:
                chunks.append(f"- [{source}] {chunk[:300]}")
        if chunks:
            rag_section = f"Relevant documentation:\n" + "\n".join(chunks) + "\n\n"

    user_prompt = (
        f"Task:\n{restructured_query}\n\n"
        f"Plan:\n{current_plan}\n\n"
        f"Codebase snapshot:\n{json.dumps(code_base, default=str)[:5000]}\n\n"
        f"{rag_section}"
        f"Available MCP tools:\n{tools_summary}\n\n"
        "Start executing now."
    )
    return system_prompt, user_prompt
