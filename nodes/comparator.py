"""
Comparator Node
"""

import logging
from core.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
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
