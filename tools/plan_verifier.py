"""
Plan Verifier Tool
Scores the plan quality using LLM as judge (0.0 = bad, 1.0 = perfect).
Used by: plan_node
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage
from providers.provider import get_llm
from prompts.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


def verify_plan(plan: str, restructured_query: str, code_base: dict, config: dict = None) -> dict:
    """Score the plan quality using LLM as judge. Returns {"score": float, "feedback": str}."""
    llm = get_llm(config)

    # Render the POML prompt with context
    prompt_content = render_prompt("plan_verifier", {
        "restructured_query": restructured_query,
        "code_base": json.dumps(code_base, default=str) if code_base else "",
        "plan": plan,
    })

    response = llm.invoke([HumanMessage(content=prompt_content)])

    return _parse_judge_response(response.content)


def _parse_judge_response(text: str) -> dict:
    """Parse LLM judge response, with regex fallback."""
    # Try JSON parse first
    try:
        data = json.loads(text)
        return {
            "score": float(data["score"]),
            "feedback": str(data.get("feedback", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Try extracting JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return {
                "score": float(data["score"]),
                "feedback": str(data.get("feedback", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Regex fallback: extract score
    score_match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', text)
    feedback_match = re.search(r'"?feedback"?\s*[:=]\s*"([^"]*)"', text)

    score = float(score_match.group(1)) if score_match else 0.0
    feedback = feedback_match.group(1) if feedback_match else text[:200]

    return {"score": min(max(score, 0.0), 1.0), "feedback": feedback}
