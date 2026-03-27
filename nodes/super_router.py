"""
Super Router Node
Classifies the task as "simple" or "complex" using LLM-based analysis.
- simple: small coding task → implement directly via MCP tools
- complex: multi-step coding task → plan → approve → implement

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

logger = logging.getLogger(__name__)


def _classify_task(query: str, code_base: dict, config: dict) -> str:
    """Classify the task as 'simple' or 'complex'.
    Uses a direct, concise prompt that works well with smaller local models."""
    llm = get_llm(config)

    # Count files in the codebase for context
    file_count = len(code_base.get("file_tree", [])) if code_base else 0

    # Use a very concise prompt that small models handle better
    prompt = (
        f'Classify this coding task as "simple" or "complex".\n\n'
        f'Task: {query}\n'
        f'Project has {file_count} files.\n\n'
        f'Rules:\n'
        f'- simple = pure logic, single script, no UI, no external APIs, no frameworks '
        f'(e.g. "add two numbers", "sort a list", "fizzbuzz")\n'
        f'- complex = uses frameworks (Flask, Streamlit, Django, FastAPI), builds an app, '
        f'has UI, uses APIs, multiple components, or anything non-trivial\n\n'
        f'Answer with one word only: simple or complex'
    )

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()
    logger.info("Super router raw LLM response: %s", raw[:200])

    # Check first word for classification
    first_word = raw.split()[0].rstrip(".,;:!") if raw.split() else ""

    if first_word == "simple":
        return "simple"
    if first_word == "complex":
        return "complex"

    # Fallback: check if classification word appears anywhere
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
    error_log = list(state.get("error_log", []) or [])

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
            "error_log": error_log,
        }

    # Re-entry from clarification: route_type already forced to "complex"
    if existing_route == "complex":
        logger.info("Super router: route_type already 'complex' (from clarification), skipping classification.")
        return {
            "route_type": "complex",
            "active_provider": provider["name"],
            "active_model": provider["model"],
            "current_node": "super_router",
            "error_log": error_log,
        }

    # Fresh task — classify with LLM
    route_type = _classify_task(query, code_base, config)
    logger.info("Super router classified task as: %s", route_type)

    return {
        "route_type": route_type,
        "active_provider": provider["name"],
        "active_model": provider["model"],
        "current_node": "super_router",
        "error_log": error_log,
    }
