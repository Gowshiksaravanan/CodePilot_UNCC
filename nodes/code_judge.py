"""
Code Judge Node
Reads: files_modified, current_plan, restructured_query, execution_log, rag_query_results
Writes: implementation_status, current_node, error_log
Uses LLM to verify that the implementation correctly follows the plan.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm

logger = logging.getLogger(__name__)


def _parse_judge_response(text: str) -> dict:
    """Parse LLM judge response into {status, feedback}, with fallbacks."""
    # Try direct JSON parse
    try:
        data = json.loads(text)
        return {
            "status": str(data.get("status", "incorrect")).lower(),
            "feedback": str(data.get("feedback", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return {
                "status": str(data.get("status", "incorrect")).lower(),
                "feedback": str(data.get("feedback", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Regex fallback
    status_match = re.search(r'"?status"?\s*[:=]\s*"?(correct|incorrect)"?', text, re.IGNORECASE)
    feedback_match = re.search(r'"?feedback"?\s*[:=]\s*"([^"]*)"', text)

    status = status_match.group(1).lower() if status_match else "incorrect"
    feedback = feedback_match.group(1) if feedback_match else text[:300]

    return {"status": status, "feedback": feedback}


def run(state: AgentState) -> dict:
    """Verify implementation correctness using LLM judge."""
    config = load_config()
    error_log = list(state.get("error_log", []) or [])

    execution_log: List[Dict[str, Any]] = list(state.get("execution_log", []) or [])
    files_modified: List[str] = list(state.get("files_modified", []) or [])
    plan: str = state.get("current_plan", "") or ""
    query: str = state.get("restructured_query", "") or ""
    rag_results = state.get("rag_query_results", []) or []

    # Quick rule-based checks before invoking LLM
    if not execution_log:
        return {
            "implementation_status": {"status": "incorrect", "feedback": "No tool calls were executed."},
            "current_node": "code_judge",
            "error_log": error_log,
        }

    # Simple → Ollama, Complex → cloud
    try:
        route_type = state.get("route_type", "complex")
        if route_type == "complex":
            llm = get_llm(config, route_type=route_type, force_cloud=True)
        else:
            llm = get_llm(config, route_type="simple")

        # Prepare execution log summary (truncate to avoid token overflow)
        log_summary = json.dumps(execution_log[:20], default=str)[:4000]
        files_str = ", ".join(files_modified) if files_modified else "No files recorded as modified"
        rag_str = json.dumps(rag_results[:3], default=str)[:1000] if rag_results else "None"

        # For simple tasks with no plan, use a simpler evaluation
        if not plan or plan.strip() == "":
            plan = f"(No formal plan — simple task) Write code to: {query}"

        # Use inline prompt instead of POML to avoid XML parse errors
        # from execution log content containing <, >, / characters
        prompt_content = (
            "You are a strict code implementation judge.\n"
            "Review the implementation below and determine if it is correct.\n\n"
            f"Original Task: {query}\n\n"
            f"Plan: {plan}\n\n"
            f"Files Modified: {files_str}\n\n"
            f"Execution Log:\n{log_summary}\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"status": "correct" or "incorrect", "feedback": "explanation"}\n'
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt_content)])
        except Exception as llm_err:
            # Rate limit fallback to Ollama
            err_str = str(llm_err)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                logger.warning("Cloud API rate limited in code_judge, falling back to Ollama")
                llm = get_llm(config, route_type="simple")
                response = llm.invoke([HumanMessage(content=prompt_content)])
            else:
                raise
        result = _parse_judge_response(response.content)

        logger.info("Code judge verdict: %s", result["status"])

        return {
            "implementation_status": result,
            "current_node": "code_judge",
            "error_log": error_log,
        }

    except Exception as exc:
        logger.exception("Code judge LLM call failed, falling back to rule-based check")
        error_log.append({
            "node": "code_judge",
            "tool": "llm_invoke",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Fallback: rule-based check
        errors = [e for e in execution_log if e.get("status") == "error"]
        if errors:
            last = errors[-1]
            feedback = f"LLM judge unavailable. Rule-based: tool call failed — {last.get('tool')}: {last.get('output')}"
            return {
                "implementation_status": {"status": "incorrect", "feedback": feedback},
                "current_node": "code_judge",
                "error_log": error_log,
            }

        return {
            "implementation_status": {"status": "correct", "feedback": "LLM judge unavailable; rule-based check passed."},
            "current_node": "code_judge",
            "error_log": error_log,
        }
