"""
Comparator Node
Reads: context (previous), code_base
Writes: context_difference, has_prev_context
"""

import logging
from core.state import AgentState

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.85
MAX_ITERATIONS = 3


def run(state: AgentState) -> dict:
    score = state.get("plan_score", 0.0)
    iteration = state.get("plan_iteration_count", 0)
    feedback = state.get("plan_feedback", "")

    logger.info(f"Comparator: score={score}, iteration={iteration}")

    # Case 1: Good plan -> proceed
    if score >= SCORE_THRESHOLD:
        logger.info("Plan accepted.")
        return {
            "route_type": "simple",  # or move to implement node
            "needs_clarification": False,
            "current_node": "comparator"
        }

    # Case 2: Retry planning
    if iteration < MAX_ITERATIONS:
        logger.info("Retrying plan generation.")
        return {
            "route_type": "complex",
            "needs_clarification": False,
            "current_node": "comparator"
        }

    # Case 3: Too many failures -> ask user
    logger.info("Max iterations reached. Asking for clarification.")
    return {
        "route_type": "complex",
        "needs_clarification": True,
        "current_node": "comparator"
    }
