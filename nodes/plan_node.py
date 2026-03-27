"""
Plan Node
Reads: restructured_query, code_base, context, user_feedback, plan_feedback, user_responses,
       plan_iteration_count, plan_history, rag_query_results
Writes: current_plan, plan_score, plan_feedback, plan_iteration_count, plan_history,
        current_node, rag_query_results, rag_fallback_used, rag_fallback_results
Uses: plan_generator tool, plan_verifier tool
"""

import logging
from typing import List, Tuple
import asyncio
import json

from core.state import AgentState
from core.config import load_config
from mcp_client.client import MCPClient
from tools.plan_generator import generate_plan
from tools.plan_verifier import verify_plan

logger = logging.getLogger(__name__)


def _parse_mcp_text_json(tool_result) -> dict:
    """
    MCP returns a CallToolResult whose content usually contains a single TextContent block.
    """
    texts = []
    for block in getattr(tool_result, "content", []) or []:
        if hasattr(block, "text"):
            texts.append(block.text)
    joined = "\n".join(texts).strip()
    if not joined:
        return {}
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"raw": joined}


async def _query_rag_and_fallback(query: str, config: dict) -> Tuple[List[dict], bool, List[dict]]:
    """
    Per design: query Custom RAG first; if insufficient, fall back to Tavily web search.
    Returns (rag_results, rag_fallback_used, rag_fallback_results).
    """
    client = MCPClient(config)
    await client.connect_all(only_servers=["rag", "tavily"])

    rag_results: list[dict] = []
    fallback_used = False
    fallback_results: list[dict] = []

    try:
        rag_res = await client.call_tool("query_python_docs", {"query": query})
        rag_payload = _parse_mcp_text_json(rag_res)
        rag_results = list(rag_payload.get("results") or [])
    except Exception:
        rag_results = []

    # "Sufficient?" heuristic from design: non-empty and scores > 0.7
    sufficient = False
    if rag_results:
        try:
            sufficient = any(float(r.get("score", 0.0)) >= 0.7 for r in rag_results)
        except Exception:
            sufficient = True

    if not sufficient:
        fallback_used = True
        try:
            tav_res = await client.call_tool("tavily_search", {"query": query})
            tav_payload = _parse_mcp_text_json(tav_res)
            # Tavily MCP typically returns a list of results; keep as-is in state
            fallback_results = list(tav_payload.get("results") or tav_payload.get("data") or [])
        except Exception:
            fallback_results = []

    await client.cleanup()
    return rag_results, fallback_used, fallback_results


def run(state: AgentState) -> dict:
    """Generate an execution plan and score it with LLM judge."""
    config = load_config()
    error_log = list(state.get("error_log", []) or [])

    # Read state fields
    query = state["restructured_query"]
    code_base = state.get("code_base", {})
    context = state.get("context", {})
    user_feedback = state.get("user_feedback", "")
    user_responses = state.get("user_responses", [])
    plan_feedback = state.get("plan_feedback", "")
    plan_history = list(state.get("plan_history", []))
    iteration = state.get("plan_iteration_count", 0)
    route_type = state.get("route_type", "")

    # Query RAG for Python docs (fallback to Tavily web search if insufficient)
    try:
        from core.async_utils import run_async
        rag_results, rag_fallback_used, rag_fallback_results = run_async(
            _query_rag_and_fallback(query, config)
        )
    except Exception as exc:
        logger.exception("RAG query failed in plan_node")
        from datetime import datetime, timezone
        error_log.append({
            "node": "plan_node",
            "tool": "rag_query",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        rag_results, rag_fallback_used, rag_fallback_results = [], False, []

    # Generate plan
    try:
        plan = generate_plan(
            restructured_query=query,
            code_base=code_base,
            context=context,
            user_responses=user_responses if user_responses else None,
            user_feedback=user_feedback,
            plan_feedback=plan_feedback,
            rag_context=rag_results if rag_results else None,
            config=config,
            route_type=route_type,
        )
    except Exception as exc:
        logger.exception("Plan generation failed")
        from datetime import datetime, timezone
        error_log.append({
            "node": "plan_node",
            "tool": "generate_plan",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        plan = state.get("current_plan", "") or "Plan generation failed."

    # Verify plan with LLM judge
    try:
        result = verify_plan(plan, query, code_base, config=config, route_type=route_type)
        score = result["score"]
        feedback = result["feedback"]
    except Exception as exc:
        logger.exception("Plan verification failed")
        from datetime import datetime, timezone
        error_log.append({
            "node": "plan_node",
            "tool": "verify_plan",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        score = 0.0
        feedback = f"Plan verification failed: {exc}"

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
        "rag_fallback_used": rag_fallback_used,
        "rag_fallback_results": rag_fallback_results,
        "error_log": error_log,
    }
