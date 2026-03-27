"""
Clarification Generator Tool
Generates clarifying questions when the LLM judge rejects a plan.
Used by: user_clarification
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage
from providers.provider import get_llm

logger = logging.getLogger(__name__)


def generate_clarifications(plan: str, plan_feedback: str, plan_score: float, config: dict = None, route_type: str = None) -> list:
    """Generate clarification questions based on judge feedback. Returns list of question strings."""
    llm = get_llm(config, route_type=route_type)

    # Use inline prompt to avoid POML XML parse errors from dynamic content
    prompt_content = (
        "You are a coding assistant that analyzes rejected execution plans to identify\n"
        "ambiguities, missing information, and assumptions that need user confirmation.\n"
        "Generate precise, targeted clarification questions that will resolve the\n"
        "specific issues the plan reviewer identified.\n\n"
        "QUESTION GENERATION RULES:\n"
        "- Generate 2-5 questions — enough to cover all ambiguities.\n"
        "- Each question must address a SPECIFIC issue from the reviewer's feedback.\n"
        "- Questions should be about user PREFERENCES and REQUIREMENTS, not implementation details.\n"
        "- Do NOT ask questions the codebase already answers.\n"
        "- Do NOT ask yes/no questions when the user needs to choose between approaches.\n"
        "- DO ask about: technology choices, scope decisions, security requirements, feature behavior.\n\n"
        f"REJECTED PLAN:\n{plan}\n\n"
        f"REVIEWER FEEDBACK:\n{plan_feedback}\n\n"
        f"REVIEWER SCORE: {plan_score}\n\n"
        "Return ONLY a valid JSON array of question strings.\n"
        "Each question should be a complete sentence ending with a question mark.\n"
        "No additional text, explanation, or markdown — just the JSON array.\n"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_content)])
    except Exception as llm_err:
        err_str = str(llm_err)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            logger.warning("Cloud API rate limited in clarification_generator, falling back to Ollama")
            llm = get_llm(config, route_type="simple")
            response = llm.invoke([HumanMessage(content=prompt_content)])
        else:
            raise

    return _parse_questions(response.content)


def _parse_questions(text: str) -> list:
    """Parse LLM response into a list of question strings."""
    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(q) for q in result]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from markdown code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return [str(q) for q in result]
        except json.JSONDecodeError:
            pass

    # Fallback: split by newlines and filter lines that look like questions
    lines = [line.strip().lstrip("0123456789.-) ") for line in text.strip().split("\n")]
    questions = [line for line in lines if line and line.endswith("?")]
    return questions if questions else [text.strip()]
