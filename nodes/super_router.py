"""
Super Router Node
Classifies the task as "simple" or "complex" using LLM-based analysis.
On re-entry (after user approval/rejection), preserves the existing route_type
and lets the conditional edge (route_type_edge) handle routing via user_approved flag.

Reads: restructured_query, code_base, user_approved, route_type
Writes: route_type, active_provider, active_model, current_node
"""

import json
import logging

from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm
from prompts.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


def _classify_task(query: str, code_base: dict, config: dict) -> str:
    """Use LLM to classify the task as 'simple' or 'complex'."""
    llm = get_llm(config)

    prompt_content = render_prompt("super_router", {
        "restructured_query": query,
        "code_base": json.dumps(code_base, default=str) if code_base else "No codebase information available.",
    })

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt_content)])
    raw = response.content.strip().lower()

    # Extract classification — LLM should return just "simple" or "complex"
    if "simple" in raw:
        return "simple"
    return "complex"


def run(state: AgentState) -> dict:
    """
    Super Router: classifies task complexity and sets routing fields.

    If user_approved is already set (re-entry from approval/rejection),
    preserve the existing route_type — the conditional edge handles
    routing based on user_approved flag.

    If route_type is already "complex" (set by clarification node),
    skip LLM classification and keep it as complex.
    """
    config = load_config()
    provider = config["provider"]

    query = state.get("restructured_query", "")
    code_base = state.get("code_base", {})
    user_approved = state.get("user_approved")
    existing_route = state.get("route_type", "")

    # Re-entry from approval/rejection: don't reclassify
    if user_approved is not None:
        logger.info(
            "Super router re-entry: user_approved=%s, keeping route_type='%s'",
            user_approved, existing_route,
        )
        return {
            "active_provider": provider["name"],
            "active_model": provider["model"],
            "current_node": "super_router",
        }

    # Re-entry from clarification: route_type already forced to "complex"
    if existing_route == "complex":
        logger.info("Super router: route_type already 'complex' (from clarification), skipping classification.")
        return {
            "route_type": "complex",
            "active_provider": provider["name"],
            "active_model": provider["model"],
            "current_node": "super_router",
        }

    # Fresh task — classify with LLM
    route_type = _classify_task(query, code_base, config)
    logger.info("Super router classified task as: %s", route_type)

    return {
        "route_type": route_type,
        "active_provider": provider["name"],
        "active_model": provider["model"],
        "current_node": "super_router",
    }
