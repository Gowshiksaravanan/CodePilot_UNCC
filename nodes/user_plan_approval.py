"""
User Plan Approval Node
Reads: current_plan, plan_score
Writes: user_approved, user_feedback, current_node
Displays the plan to the user via Rich console and collects approval/rejection.
If rejected, a reason is mandatory.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession

from core.state import AgentState

console = Console()


def run(state: AgentState) -> dict:
    """Display the plan to the user and collect approval or rejection with reason."""
    error_log = list(state.get("error_log", []) or [])
    current_plan = state.get("current_plan", "")
    plan_score = state.get("plan_score", 0.0)
    plan_iteration_count = state.get("plan_iteration_count", 0)

    # Display the plan
    console.print()
    console.print(Panel(
        Markdown(current_plan),
        title="[bold cyan]Execution Plan[/bold cyan]",
        subtitle=f"Score: {plan_score:.2f} | Iteration: {plan_iteration_count}",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # Prompt for approval
    session = PromptSession()

    while True:
        console.print("[bold]Do you approve this plan?[/bold] ([green]yes[/green] / [red]no[/red])")
        try:
            response = session.prompt("❯ ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            response = "no"

        if response in ("yes", "y"):
            console.print("[green]Plan approved.[/green]\n")
            return {
                "user_approved": True,
                "user_feedback": "",
                "current_node": "user_plan_approval",
                "error_log": error_log,
            }

        if response in ("no", "n"):
            # Rejection reason is mandatory
            console.print("[yellow]Please provide a reason for rejection:[/yellow]")
            while True:
                try:
                    reason = session.prompt("❯ ").strip()
                except (KeyboardInterrupt, EOFError):
                    reason = ""

                if reason:
                    break
                console.print("[red]Reason is required. Please explain what needs to change:[/red]")

            console.print(f"[red]Plan rejected.[/red] Reason: {reason}\n")
            return {
                "user_approved": False,
                "user_feedback": reason,
                "current_node": "user_plan_approval",
                "error_log": error_log,
            }

        console.print("[dim]Please answer 'yes' or 'no'.[/dim]")
