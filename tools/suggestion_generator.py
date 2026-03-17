"""
Suggestion Generator Tool
Generates multiple-choice suggestions for each clarification question.
Used by: user_clarification
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage
from providers.provider import get_llm
from prompts.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


def generate_suggestions(questions: list, config: dict = None) -> list:
    """Generate suggestion options for each clarification question. Returns list of list of strings."""
    llm = get_llm(config)

    # Render the POML prompt with context
    prompt_content = render_prompt("suggestion_generator", {
        "questions": json.dumps(questions),
    })

    response = llm.invoke([HumanMessage(content=prompt_content)])

    return _parse_suggestions(response.content, len(questions))


def _parse_suggestions(text: str, expected_count: int) -> list:
    """Parse LLM response into list of list of strings."""
    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(item, list) for item in result):
            return [[str(s) for s in group] for group in result]
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list) and all(isinstance(item, list) for item in result):
                return [[str(s) for s in group] for group in result]
        except json.JSONDecodeError:
            pass

    # Fallback: empty suggestions for each question
    return [[] for _ in range(expected_count)]
