"""
Query Reconstruction Node
"""

import logging
import uuid
from datetime import datetime, timezone

from core.state import AgentState
from core.config import load_config
from providers.provider import get_llm
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


def run(state: AgentState) -> dict:
    """
    Reconstruct user_query into a clearer, actionable restructured_query.
    """
    config = load_config()
    llm = get_llm(config)
    error_log = list(state.get("error_log", []) or [])

    user_query = state.get("user_query", "")
    code_base = state.get("code_base", {})
    existing_context = state.get("context")

    code_base_str = str(code_base)[:2000] if code_base else "No codebase context available."

    # Use inline prompt to avoid POML XML parse errors from dynamic content
    prompt = (
        "You are a query reconstruction engine for an autonomous AI coding assistant.\n"
        "Rewrite the user's request into a clear, structured, and actionable instruction.\n\n"
        "GUIDELINES:\n"
        "- Preserve the original intent exactly.\n"
        "- Remove ambiguity and vagueness.\n"
        "- If the user is asking a QUESTION, keep it as a question. Do NOT turn it into a task.\n"
        "- If the task involves code, mention relevant components (files, functions, APIs).\n"
        "- Do NOT generate step-by-step instructions — that is the planner's job.\n"
        "- Keep the output concise but informative.\n\n"
        f"USER QUERY:\n{user_query}\n\n"
        f"CODEBASE CONTEXT:\n{code_base_str}\n\n"
        "Return ONLY the reconstructed query as a clean paragraph. No explanations or extra text.\n"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    reconstructed = response.content.strip()

    logger.info("Query reconstructed.")

    session_id = state.get("session_id") or f"session-{uuid.uuid4().hex[:12]}"
    time_started = state.get("time_started") or datetime.now(timezone.utc).isoformat()

    return {
        "restructured_query": reconstructed,
        "has_prev_context": bool(existing_context),
        "current_node": "query_reconstruction",
        "session_id": session_id,
        "time_started": time_started,
        "error_log": error_log,
    }
