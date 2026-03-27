"""
Direct Response Node
Handles knowledge/explanation questions that don't require code changes.
Uses the LLM to generate a direct answer and returns it as the final output.

Reads: restructured_query, code_base
Writes: final_response, current_node, error_log
"""

import logging

from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    """Generate a direct answer to a knowledge/explanation question."""
    config = load_config()
    error_log = list(state.get("error_log", []) or [])

    query = state.get("restructured_query", state.get("user_query", ""))
    code_base = state.get("code_base", {})

    llm = get_llm(config, route_type="simple")

    context_hint = ""
    if code_base:
        context_hint = (
            "\n\nThe user is working on a project with the following codebase context. "
            "Use it to make your answer more relevant if applicable:\n"
            f"{str(code_base)[:2000]}"
        )

    prompt = (
        f"You are a helpful coding assistant. Answer the following question clearly and concisely."
        f"{context_hint}\n\n"
        f"Question: {query}"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
    except Exception as exc:
        logger.exception("Direct response generation failed")
        answer = f"Sorry, I couldn't generate an answer: {exc}"
        from datetime import datetime, timezone
        error_log.append({
            "node": "direct_response",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    logger.info("Direct response generated.")

    return {
        "final_response": answer,
        "current_node": "direct_response",
        "error_log": error_log,
    }
