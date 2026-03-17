"""
Plan Generator Tool
Generates a step-by-step execution plan from the restructured query and codebase context.
Used by: plan_node
"""

import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from providers.provider import get_llm
from prompts.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


def generate_plan(
    restructured_query: str,
    code_base: dict,
    context: dict,
    user_responses: list = None,
    user_feedback: str = "",
    plan_feedback: str = "",
    rag_context: list = None,
    config: dict = None,
) -> str:
    """Generate an execution plan using the LLM with POML-rendered prompt."""
    llm = get_llm(config)

    # Render the POML prompt with context
    prompt_content = render_prompt("plan_generator", {
        "restructured_query": restructured_query,
        "code_base": json.dumps(code_base, default=str) if code_base else "",
        "context": json.dumps(context, default=str) if context else "",
        "plan_feedback": plan_feedback,
        "user_feedback": user_feedback,
        "user_responses": json.dumps(user_responses) if user_responses else "",
        "rag_context": json.dumps(rag_context, default=str) if rag_context else "",
    })

    response = llm.invoke([HumanMessage(content=prompt_content)])

    return response.content
