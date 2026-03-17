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
from prompts.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


def generate_clarifications(plan: str, plan_feedback: str, plan_score: float, config: dict = None) -> list:
    """Generate clarification questions based on judge feedback. Returns list of question strings."""
    llm = get_llm(config)

    # Render the POML prompt with context
    prompt_content = render_prompt("clarification_generator", {
        "plan": plan,
        "plan_feedback": plan_feedback,
        "plan_score": str(plan_score),
    })

    response = llm.invoke([HumanMessage(content=prompt_content)])

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
