"""
Context Updator Node
Reads: (scans fresh or uses comparator output)
Writes: code_base, context, has_prev_context
Uses: codebase_learner tool
"""

import logging

from core.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    # Gather inputs
    current_plan = state.get("current_plan", "")
    plan_history = state.get("plan_history", [])
    user_responses = state.get("user_responses", [])
    plan_feedback = state.get("plan_feedback", "")
    rag_results = state.get("rag_query_results", [])

    # Build structured context
    context = {
        "latest_plan": current_plan,
        "past_plans": plan_history[-3:],  # keep last 3 only
        "user_clarifications": user_responses,
        "plan_feedback": plan_feedback,
        "relevant_docs": rag_results[:5] if rag_results else [],
    }

    logger.info("Context updated with latest state.")

    return {
        "context": context,
        "current_node": "context_updator"
    }
