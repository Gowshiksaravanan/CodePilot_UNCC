"""
Plan Node
Reads: restructured_query, code_base, context, user_feedback, plan_feedback, user_responses,
       plan_iteration_count, plan_history, rag_query_results
Writes: current_plan, plan_score, plan_feedback, plan_iteration_count, plan_history,
        current_node, rag_query_results, rag_fallback_used, rag_fallback_results
Uses: plan_generator tool, plan_verifier tool
"""

import logging
from typing import List

from core.state import AgentState
from core.config import load_config
from tools.plan_generator import generate_plan
from tools.plan_verifier import verify_plan

logger = logging.getLogger(__name__)


def _query_rag(query: str) -> List[dict]:
    """Placeholder for RAG MCP server query.
    TODO: call mcp_client.call_tool("query_python_docs", {"query": query})
    Returns list of doc chunks: [{"chunk": str, "source": str, "score": float}, ...]
    """
    return []


def run(state: AgentState) -> dict:
    """Generate an execution plan and score it with LLM judge."""
    config = load_config()

    # Read state fields
    query = state["restructured_query"]
    code_base = state.get("code_base", {})
    context = state.get("context", {})
    user_feedback = state.get("user_feedback", "")
    user_responses = state.get("user_responses", [])
    plan_feedback = state.get("plan_feedback", "")
    plan_history = list(state.get("plan_history", []))
    iteration = state.get("plan_iteration_count", 0)

    # Query RAG for Python docs if needed (placeholder)
    rag_results = _query_rag(query)

    # Generate plan
    plan = generate_plan(
        restructured_query=query,
        code_base=code_base,
        context=context,
        user_responses=user_responses if user_responses else None,
        user_feedback=user_feedback,
        plan_feedback=plan_feedback,
        rag_context=rag_results if rag_results else None,
        config=config,
    )

    # Verify plan with LLM judge
    result = verify_plan(plan, query, code_base, config=config)
    score = result["score"]
    feedback = result["feedback"]

    # Update history
    plan_history.append(plan)

    logger.info("Plan node iteration %d — score: %.2f", iteration + 1, score)

    return {
        "current_plan": plan,
        "plan_score": score,
        "plan_feedback": feedback,
        "plan_iteration_count": iteration + 1,
        "plan_history": plan_history,
        "current_node": "plan_node",
        "rag_query_results": rag_results,
        "rag_fallback_used": False,
        "rag_fallback_results": [],
    }
