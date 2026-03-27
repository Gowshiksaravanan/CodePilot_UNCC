"""
Context Updator Node
"""

import logging

from core.state import AgentState
from tools.codebase_learner import scan_codebase, scan_codebase_via_mcp

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    error_log = list(state.get("error_log", []) or [])
    prev_context = state.get("context", {})
    existing_code_base = state.get("code_base", {}) or {}

    current_plan = state.get("current_plan", "")
    plan_history = state.get("plan_history", [])
    user_responses = state.get("user_responses", [])
    user_feedback = state.get("user_feedback", "")
    user_approved = state.get("user_approved", None)
    plan_feedback = state.get("plan_feedback", "")
    rag_results = state.get("rag_query_results", [])
    context_difference = state.get("context_difference", "")

    # Refresh codebase snapshot (prefer MCP filesystem scan, fallback to local)
    if not existing_code_base:
        from core.config import load_config
        from datetime import datetime, timezone

        config = load_config()
        try:
            code_base = scan_codebase_via_mcp(config)
            logger.info("Scanned codebase (%s) (%d files)", code_base.get("source"), len(code_base.get("file_tree", [])))
        except Exception as exc:
            logger.exception("MCP codebase scan failed in context_updator")
            error_log.append({
                "node": "context_updator",
                "tool": "scan_codebase_via_mcp",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            code_base = {}
    else:
        code_base = existing_code_base

    context = {
        **prev_context,
        "latest_plan": current_plan,
        "past_plans": plan_history[-3:],
        "user_clarifications": prev_context.get("user_clarifications", []) + user_responses,
        "user_feedback": user_feedback,
        "user_approved": user_approved,
        "plan_feedback": plan_feedback,
        "context_difference": context_difference,
        "relevant_docs": rag_results[:5] if rag_results else [],
        "code_base": code_base,
    }

    logger.info("Context updated and persisted successfully.")

    return {
        "context": context,
        "code_base": code_base,
        "has_prev_context": True,
        "current_node": "context_updator",
        "error_log": error_log,
    }
