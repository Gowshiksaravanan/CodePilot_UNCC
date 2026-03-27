"""
User Clarification Node
Reads: current_plan, plan_feedback, plan_score
Writes: clarification_questions, suggestions, user_responses, route_type, current_node
Uses: clarification_generator tool, suggestion_generator tool

When the LLM judge rejects the plan (score < 0.85), this node generates
clarification questions + suggestions, displays them to the user, and
collects their responses. Sets route_type='complex' so the flow returns
through comparator → context_updator → super_router → plan_node.
"""

import logging

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession

from core.state import AgentState
from core.config import load_config
from tools.clarification_generator import generate_clarifications
from tools.suggestion_generator import generate_suggestions

logger = logging.getLogger(__name__)
console = Console()


def run(state: AgentState) -> dict:
    """Generate clarification questions, display with suggestions, collect user responses."""
    config = load_config()
    error_log = list(state.get("error_log", []) or [])

    current_plan = state.get("current_plan", "")
    plan_feedback = state.get("plan_feedback", "")
    plan_score = state.get("plan_score", 0.0)
    route_type = state.get("route_type", "")

    # Generate clarification questions from judge feedback
    try:
        questions = generate_clarifications(current_plan, plan_feedback, plan_score, config=config, route_type=route_type)
    except Exception as exc:
        logger.exception("Clarification generation failed")
        from datetime import datetime, timezone
        error_log.append({
            "node": "user_clarification",
            "tool": "generate_clarifications",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        questions = [f"The plan reviewer scored this {plan_score:.2f}. What changes would you like?"]

    # Generate suggestions for each question
    try:
        suggestions = generate_suggestions(questions, config=config, route_type=route_type)
    except Exception as exc:
        logger.exception("Suggestion generation failed")
        from datetime import datetime, timezone
        error_log.append({
            "node": "user_clarification",
            "tool": "generate_suggestions",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        suggestions = [[] for _ in questions]

    # Display judge feedback to user
    console.print()
    console.print(Panel(
        f"[yellow]The plan reviewer found issues (score: {plan_score:.2f}):[/yellow]\n\n"
        f"{plan_feedback}",
        title="[bold yellow]Plan Needs Clarification[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))
    console.print()

    # Collect user responses for each question
    session = PromptSession()
    user_responses = []

    for i, question in enumerate(questions):
        console.print(f"[bold cyan]Q{i + 1}:[/bold cyan] {question}")

        # Show suggestions if available
        question_suggestions = suggestions[i] if i < len(suggestions) else []
        if question_suggestions:
            for j, suggestion in enumerate(question_suggestions):
                console.print(f"  [dim]{chr(97 + j)})[/dim] {suggestion}")
            console.print(f"  [dim]Or type your own answer.[/dim]")

        # Get user response
        while True:
            try:
                response = session.prompt("❯ ").strip()
            except (KeyboardInterrupt, EOFError):
                response = ""

            if response:
                # If user typed a letter matching a suggestion, use the suggestion text
                if (
                    len(response) == 1
                    and response.isalpha()
                    and question_suggestions
                ):
                    idx = ord(response.lower()) - ord("a")
                    if 0 <= idx < len(question_suggestions):
                        response = question_suggestions[idx]
                        console.print(f"  [green]→ {response}[/green]")

                user_responses.append(response)
                break

            console.print("  [red]Please provide an answer.[/red]")

        console.print()

    console.print("[green]Responses collected. Regenerating plan...[/green]\n")

    return {
        "clarification_questions": questions,
        "suggestions": suggestions,
        "user_responses": user_responses,
        "route_type": "complex",
        "current_node": "user_clarification",
        "error_log": error_log,
    }
