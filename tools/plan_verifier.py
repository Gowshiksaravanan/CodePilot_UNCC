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

logger = logging.getLogger(__name__)


def verify_plan(plan: str, restructured_query: str, code_base: dict, config: dict = None, route_type: str = None) -> dict:
    """Score the plan quality using LLM as judge. Returns {"score": float, "feedback": str}."""
    llm = get_llm(config, route_type=route_type)

    code_base_str = json.dumps(code_base, default=str)[:3000] if code_base else "No codebase info"

    # Use inline prompt to avoid POML XML parse errors from dynamic content
    prompt_content = (
        "You are a strict, detail-oriented plan quality judge for software development tasks.\n"
        "Evaluate execution plans and score them on correctness, completeness, and actionability.\n\n"
        "EVALUATION CRITERIA:\n"
        "- Completeness: Does the plan cover ALL aspects of the task?\n"
        "- File path accuracy: Do referenced paths exist or follow codebase conventions?\n"
        "- Step ordering: Are dependencies installed before code that uses them?\n"
        "- Atomicity: Is each step a single, concrete action?\n"
        "- Error handling: For I/O, network, auth tasks, are error cases handled?\n"
        "- Verification: Does the plan end with a verification step?\n\n"
        "AMBIGUITY DETECTION (Critical):\n"
        "- Every step must be specific enough for an automated agent to execute without guessing.\n"
        "- Flag vague language like 'add appropriate', 'handle as needed', 'configure properly'.\n"
        "- Technology choices must be explicit.\n"
        "- Each ambiguous step reduces score by 0.05-0.10.\n\n"
        "ASSUMPTION DETECTION (Critical):\n"
        "- Compare every file path and package against the codebase info.\n"
        "- Flag when the plan assumes technology not requested by user or present in codebase.\n"
        "- Flag when plan chooses between valid approaches without user specifying preference.\n"
        "- Each unwarranted assumption reduces score by 0.05-0.15.\n\n"
        "SCORING:\n"
        "- 0.90-1.00: Excellent — complete, correct, ready for execution.\n"
        "- 0.75-0.89: Good — minor issues.\n"
        "- 0.60-0.74: Needs improvement — missing important steps or wrong assumptions.\n"
        "- 0.40-0.59: Poor — significant gaps.\n"
        "- Below 0.40: Rejected — wrong approach entirely.\n\n"
        f"ORIGINAL TASK:\n{restructured_query}\n\n"
        f"CODEBASE INFORMATION:\n{code_base_str}\n\n"
        f"PLAN TO EVALUATE:\n{plan}\n\n"
        "Respond with ONLY a single line of valid JSON:\n"
        '{"score": 0.XX, "feedback": "detailed feedback explaining all issues found"}\n'
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_content)])
    except Exception as llm_err:
        err_str = str(llm_err)
        if "rate_limit" in err_str.lower() or "429" in err_str:
            logger.warning("Cloud API rate limited in plan_verifier, falling back to Ollama")
            llm = get_llm(config, route_type="simple")
            response = llm.invoke([HumanMessage(content=prompt_content)])
        else:
            raise

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

    # Regex fallback: extract score from various formats
    # Try "score": 0.85 or score: 0.85 or Score: 0.85
    score_match = re.search(r'"?score"?\s*[:=]\s*([\d.]+)', text, re.IGNORECASE)
    if not score_match:
        # Try standalone decimal like "0.85" or "85%" or "85/100"
        score_match = re.search(r'\b(0\.\d+|1\.0)\b', text)
    if not score_match:
        # Try percentage
        pct_match = re.search(r'(\d{1,3})\s*[/%]', text)
        if pct_match:
            pct = int(pct_match.group(1))
            if 0 <= pct <= 100:
                return {"score": pct / 100.0, "feedback": text[:500]}
    if not score_match:
        # Try "X out of 10" or "X/10"
        out_of_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10\b', text)
        if out_of_match:
            return {"score": min(float(out_of_match.group(1)) / 10.0, 1.0), "feedback": text[:500]}

    feedback_match = re.search(r'"?feedback"?\s*[:=]\s*"([^"]*)"', text, re.IGNORECASE)

    score = float(score_match.group(1)) if score_match else 0.5
    feedback = feedback_match.group(1) if feedback_match else text[:500]

    return {"score": min(max(score, 0.0), 1.0), "feedback": feedback}
