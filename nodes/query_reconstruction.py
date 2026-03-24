"""
Query Reconstruction Node
"""

import logging
from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm
from prompts.prompt_renderer import render_prompt
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    config = load_config()
    llm = get_llm(config)

    user_query = state.get("user_query", "")
    code_base = state.get("code_base", {})
    existing_context = state.get("context")
    files_modified = state.get("files_modified", [])
    execution_log = state.get("execution_log", [])
    has_prev_context = bool(existing_context) or bool(files_modified) or bool(execution_log)

    prompt = render_prompt("query_reconstruction", {
        "user_query": user_query,
        "code_base": str(code_base) if code_base else "No codebase context available."
    })

    response = llm.invoke([HumanMessage(content=prompt)])
    reconstructed = response.content.strip()

    logger.info("Query reconstructed.")

    return {
        "restructured_query": reconstructed,
        "has_prev_context": has_prev_context,
        "current_node": "query_reconstruction"
    }