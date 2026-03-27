"""
Comparator Node
"""

import logging

from core.state import AgentState
from tools.codebase_learner import scan_codebase

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    error_log = list(state.get("error_log", []) or [])
    prev_context = state.get("context", {})
    prev_code_base = prev_context.get("code_base", {}) or {}

    current_plan = state.get("current_plan", "")
    plan_feedback = state.get("plan_feedback", "")
    user_feedback = state.get("user_feedback", "")
    user_responses = state.get("user_responses", [])
    user_approved = state.get("user_approved", None)

    prev_plan = prev_context.get("latest_plan", "")
    prev_feedback = prev_context.get("plan_feedback", "")
    prev_user_clarifications = prev_context.get("user_clarifications", [])

    differences = []

    if current_plan and current_plan != prev_plan:
        differences.append("Execution plan has been updated.")

    if plan_feedback and plan_feedback != prev_feedback:
        differences.append(f"Plan feedback changed: {plan_feedback}")

    if user_feedback:
        differences.append(f"User feedback provided: {user_feedback}")

    if user_responses and user_responses != prev_user_clarifications:
        differences.append(f"New user clarifications: {user_responses}")

    if user_approved is not None:
        differences.append(f"User approval status: {user_approved}")

    # Refresh current codebase snapshot and compare to previous snapshot (if any)
    import os
    from datetime import datetime, timezone
    from core.config import load_config

    config = load_config()
    project_root = config.get("project_path", os.getcwd())
    try:
        current_code_base = scan_codebase(project_root)
    except Exception as exc:
        logger.exception("Codebase scan failed in comparator")
        error_log.append({
            "node": "comparator",
            "tool": "scan_codebase",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        current_code_base = state.get("code_base", {})
    if prev_code_base:
        prev_files = set(prev_code_base.get("file_tree", []) or [])
        curr_files = set(current_code_base.get("file_tree", []) or [])
        added = sorted(curr_files - prev_files)
        removed = sorted(prev_files - curr_files)
        if added:
            differences.append(f"Codebase files added: {', '.join(added[:8])}" + (" ..." if len(added) > 8 else ""))
        if removed:
            differences.append(f"Codebase files removed: {', '.join(removed[:8])}" + (" ..." if len(removed) > 8 else ""))
    else:
        differences.append("Codebase snapshot created (no previous snapshot to compare).")

    context_difference = (
        "\n".join(differences) if differences else "No significant changes detected."
    )

    logger.info("Comparator detected changes:\n%s", context_difference)

    return {
        "context_difference": context_difference,
        "code_base": current_code_base,
        "has_prev_context": True,
        "current_node": "comparator",
        "error_log": error_log,
    }
