"""
Plan Verifier Tool
Scores the plan quality using LLM as judge (0.0 = bad, 1.0 = perfect).
Used by: plan_node
"""


def verify_plan(plan: str, restructured_query: str, code_base: dict) -> dict:
    """TODO: Implement plan verification. Returns {"score": float, "feedback": str}."""
    return {"score": 0.0, "feedback": ""}
