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

logger = logging.getLogger(__name__)


def generate_suggestions(questions: list, config: dict = None, route_type: str = None) -> list:
    """Generate suggestion options for each clarification question. Returns list of list of strings."""
    llm = get_llm(config, route_type=route_type)

    questions_str = json.dumps(questions)

    # Use inline prompt to avoid POML XML parse errors from dynamic content
    prompt_content = (
        "You are a coding assistant that generates multiple-choice suggestions\n"
        "for clarification questions about software development tasks.\n\n"
        "For each question below, generate 2-4 concrete suggestion options that\n"
        "the user can choose from. Options should cover the most common/reasonable choices.\n\n"
        f"QUESTIONS:\n{questions_str}\n\n"
        "Return ONLY a valid JSON array of arrays. Each inner array contains suggestion strings\n"
        "for the corresponding question. No additional text.\n"
        'Example: [["Option A", "Option B"], ["Option X", "Option Y", "Option Z"]]\n'
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_content)])
    except Exception as llm_err:
        err_str = str(llm_err)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            logger.warning("Cloud API rate limited in suggestion_generator, falling back to Ollama")
            llm = get_llm(config, route_type="simple")
            response = llm.invoke([HumanMessage(content=prompt_content)])
        else:
            raise

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
