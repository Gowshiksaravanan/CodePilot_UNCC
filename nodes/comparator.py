"""
Comparator Node
"""

import logging
import os

from core.state import AgentState
from tools.codebase_learner import scan_codebase

logger = logging.getLogger(__name__)


def _extract_files_snapshot(codebase: dict) -> dict:
    """Accept either {files: {...}} or direct {path: metadata/content} shape."""
    if not isinstance(codebase, dict):
        return {}

    files = codebase.get("files")
    if isinstance(files, dict):
        return files

    # Fallback for direct path->value shapes.
    snapshot = {}
    for path, value in codebase.items():
        if isinstance(value, dict):
            snapshot[path] = {
                "size": value.get("size"),
                "mtime": value.get("mtime"),
            }
        elif isinstance(value, str):
            snapshot[path] = {
                "size": len(value),
                "mtime": None,
            }
    return snapshot


def _diff_snapshots(previous: dict, current: dict) -> str:
    prev_files = set(previous.keys())
    curr_files = set(current.keys())

    added = sorted(curr_files - prev_files)
    removed = sorted(prev_files - curr_files)

    modified = []
    for path in sorted(prev_files & curr_files):
        prev_meta = previous.get(path, {})
        curr_meta = current.get(path, {})
        prev_sig = (prev_meta.get("size"), prev_meta.get("mtime"))
        curr_sig = (curr_meta.get("size"), curr_meta.get("mtime"))
        if prev_sig != curr_sig:
            modified.append(path)

    parts = []
    if added:
        parts.append("New files: " + ", ".join(added[:8]) + (" ..." if len(added) > 8 else ""))
    if removed:
        parts.append("Removed files: " + ", ".join(removed[:8]) + (" ..." if len(removed) > 8 else ""))
    if modified:
        parts.append("Modified files: " + ", ".join(modified[:8]) + (" ..." if len(modified) > 8 else ""))

    return "\n".join(parts) if parts else "No significant codebase differences detected."


def run(state: AgentState) -> dict:
    # Follow-up implementation mode: compare current codebase vs previous context.
    files_modified = state.get("files_modified", [])
    execution_log = state.get("execution_log", [])
    if files_modified or execution_log:
        prev_context = state.get("context", {})

        # Support multiple keys to avoid breaking existing context shapes.
        previous_codebase = (
            prev_context.get("code_base_snapshot")
            or prev_context.get("codebase_snapshot")
            or prev_context.get("code_base")
            or {}
        )

        current_codebase = scan_codebase(os.getcwd())

        previous_snapshot = _extract_files_snapshot(previous_codebase)
        current_snapshot = _extract_files_snapshot(current_codebase)

        context_difference = _diff_snapshots(previous_snapshot, current_snapshot)

        if files_modified:
            reported = ", ".join(files_modified[:8])
            tail = " ..." if len(files_modified) > 8 else ""
            context_difference += f"\nImplementation-reported files: {reported}{tail}"

        logger.info("Comparator detected follow-up code changes:\n%s", context_difference)

        return {
            "context_difference": context_difference,
            "has_prev_context": bool(prev_context),
            "current_node": "comparator",
        }

    prev_context = state.get("context", {})

    current_plan = state.get("current_plan", "")
    plan_feedback = state.get("plan_feedback", "")
    user_feedback = state.get("user_feedback", "")
    user_responses = state.get("user_responses", [])
    user_approved = state.get("user_approved", None)

    # Previous info from context
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

    context_difference = ("\n".join(differences)
                          if differences
                          else "No significant changes detected.")

    logger.info("Comparator detected changes:\n%s", context_difference)

    return {"context_difference": context_difference,
            "has_prev_context": True, # future iterations go through comparator
            "current_node": "comparator",
    }