"""
Plan Generator Tool
Generates a step-by-step execution plan from the restructured query and codebase context.
Used by: plan_node
"""

import json
import logging

from langchain_core.messages import HumanMessage
from providers.provider import get_llm

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
    route_type: str = None,
) -> str:
    """Generate an execution plan using the LLM."""
    llm = get_llm(config, route_type=route_type)

    code_base_str = json.dumps(code_base, default=str)[:3000] if code_base else "No codebase info"
    context_str = json.dumps(context, default=str)[:2000] if context else "No context"

    # Build optional sections
    feedback_section = ""
    if plan_feedback or user_feedback:
        feedback_section = (
            "PREVIOUS PLAN FEEDBACK (address ALL points in the new plan):\n"
            f"Reviewer critique: {plan_feedback}\n"
            f"User feedback: {user_feedback}\n\n"
        )

    clarification_section = ""
    if user_responses:
        clarification_section = (
            "USER CLARIFICATION RESPONSES (incorporate their choices):\n"
            f"{json.dumps(user_responses)}\n\n"
        )

    rag_section = ""
    if rag_context:
        rag_section = (
            "DOCUMENTATION REFERENCE:\n"
            f"{json.dumps(rag_context, default=str)[:2000]}\n\n"
        )

    # Use inline prompt to avoid POML XML parse errors from dynamic content
    prompt_content = (
        "You are CodePilot, an expert AI coding assistant that generates precise, step-by-step\n"
        "execution plans for software development tasks. You produce plans that are immediately\n"
        "actionable by an automated coding agent that uses filesystem tools.\n\n"
        "PLANNING RULES:\n"
        "- Every step MUST be a single atomic action.\n"
        "- Reference exact file paths from the codebase. For new files, specify full path.\n"
        "- When modifying a file, describe precisely what to add/remove/change and where.\n"
        "- Include dependency installation steps BEFORE code that uses them.\n"
        "- Order steps logically — create base files before files that depend on them.\n"
        "- End with a verification step when applicable.\n"
        "- Do NOT include explanations or commentary — only numbered steps.\n"
        "- Do NOT assume packages are installed unless in codebase dependencies.\n\n"
        "STEP FORMAT (one action per numbered step):\n"
        "- Install: 'Install dependencies: Run `command`'\n"
        "- Create file: 'Create file `path/to/file.ext` — description of contents'\n"
        "- Modify file: 'Modify `path/to/file.ext` — what to add/change/remove and where'\n"
        "- Create directory: 'Create directory `path/to/dir/`'\n"
        "- Run command: 'Run `command` — purpose'\n\n"
        f"USER TASK:\n{restructured_query}\n\n"
        f"CODEBASE INFORMATION:\n{code_base_str}\n\n"
        f"PROJECT CONTEXT:\n{context_str}\n\n"
        f"{feedback_section}"
        f"{clarification_section}"
        f"{rag_section}"
        "Output ONLY the numbered plan. No commentary outside the steps.\n"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_content)])
    except Exception as llm_err:
        err_str = str(llm_err)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            logger.warning("Cloud API rate limited in plan_generator, falling back to Ollama")
            llm = get_llm(config, route_type="simple")
            response = llm.invoke([HumanMessage(content=prompt_content)])
        else:
            raise

    return response.content
