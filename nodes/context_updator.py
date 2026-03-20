"""
Context Updator Node
"""

import logging
from core.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    prev_context = state.get("context", {})

    current_plan = state.get("current_plan", "")
    plan_history = state.get("plan_history", [])
    user_responses = state.get("user_responses", [])
    user_feedback = state.get("user_feedback", "")
    user_approved = state.get("user_approved", None)
    plan_feedback = state.get("plan_feedback", "")
    rag_results = state.get("rag_query_results", [])
    context_difference = state.get("context_difference", "")

    # Context is updated
    context = {
        **prev_context,  # preserves existing memory
        "latest_plan": current_plan,
        "past_plans": plan_history[-3:],  # keeps recent plans
        "user_clarifications": prev_context.get("user_clarifications", []) + user_responses,
        "user_feedback": user_feedback,
        "user_approved": user_approved,
        "plan_feedback": plan_feedback,
        "context_difference": context_difference,
        "relevant_docs": rag_results[:5] if rag_results else [], # RAG context
    }

    logger.info("Context updated and persisted successfully.")

    return {
        "context": context,
        "has_prev_context": True,
        "current_node": "context_updator",
    }
