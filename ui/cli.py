#!/usr/bin/env python3
"""
CodePilot — AI-Powered CLI Coding Agent
========================================
Run:  python cli.py
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window, ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.application import Application
from prompt_toolkit.styles import Style as PTStyle
import shutil
import os
import asyncio
import sys
from typing import Dict, Any

console = Console()

WELCOME = """
 ██████╗ ██████╗ ██████╗ ███████╗██████╗ ██╗██╗      ██████╗ ████████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗██║██║     ██╔═══██╗╚══██╔══╝
██║     ██║   ██║██║  ██║█████╗  ██████╔╝██║██║     ██║   ██║   ██║
██║     ██║   ██║██║  ██║██╔══╝  ██╔═══╝ ██║██║     ██║   ██║   ██║
╚██████╗╚██████╔╝██████╔╝███████╗██║     ██║███████╗╚██████╔╝   ██║
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝ ╚═════╝   ╚═╝
"""

HELP_TEXT = """
**Commands:**
- Type any natural language instruction to get started
- `/help`  — Show this help message
- `/clear` — Clear the screen
- `/mode`  — Toggle execution mode (confirm/auto)
- `/exit`  — Quit the agent
"""

SHORTCUTS = [
    ("? ", "for shortcuts"),
    ("  /help ", "commands"),
    ("  /clear ", "reset screen"),
    ("  Ctrl+C ", "cancel"),
]


def get_status_bar():
    """Build the bottom status bar text."""
    term_width = shutil.get_terminal_size().columns
    parts = []
    for key, desc in SHORTCUTS:
        parts.append(("class:status-key", key))
        parts.append(("class:status-desc", desc))
    # Right-aligned project path
    left = "".join(k + d for k, d in SHORTCUTS)
    cwd = os.path.basename(os.getcwd())
    right_text = f"  CodePilot · {cwd}"
    parts.append(("class:status-desc", right_text))
    return parts


STATUS_STYLE = PTStyle.from_dict({
    "status-bar":  "bg:#1a1a2e #e0e0e0",
    "status-key":  "bg:#1a1a2e bold #00d4ff",
    "status-desc": "bg:#1a1a2e #888888",
    "prompt":      "bold #00d4ff",
    "separator":   "#333333",
})


def print_welcome():
    console.print(Text(WELCOME, style="bold cyan"), justify="center")
    console.print(
        Panel(
            "[bold]CodePilot[/bold] — AI-Powered CLI Coding Agent\n"
            "Type a coding instruction in natural language.\n"
            "Type [cyan]/help[/cyan] for commands or [cyan]/exit[/cyan] to quit.",
            border_style="cyan",
            padding=(1, 2),
        ),
        justify="center",
    )
    console.print()


def print_status_bar():
    """Print a styled status bar at the bottom using Rich."""
    term_width = shutil.get_terminal_size().columns
    left = ""
    segments = []
    for key, desc in SHORTCUTS:
        segments.append(f"[bold cyan]{key}[/bold cyan][dim]{desc}[/dim]")
    left_text = "  ".join(segments)

    cwd = os.path.basename(os.getcwd())
    right_text = f"[bold cyan]CodePilot[/bold cyan] [dim]· {cwd}[/dim]"

    bar = f"  {left_text}{'':>10}{right_text}  "
    console.print(bar)


def handle_command(command: str) -> bool:
    """Handle slash commands. Returns True if the main loop should continue."""
    cmd = command.strip().lower()

    if cmd in ("/exit", "/quit"):
        console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
        return False

    if cmd == "/help":
        console.print(Markdown(HELP_TEXT))
        return True

    if cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        print_welcome()
        return True

    if cmd.startswith("/mode"):
        console.print("[dim]Use:[/dim] /mode confirm  [dim]or[/dim]  /mode auto")
        return True

    console.print(f"[yellow]Unknown command:[/yellow] {command}")
    return True


def _default_state() -> Dict[str, Any]:
    return {
        "user_query": "",
        "restructured_query": "",
        "context": {},
        "code_base": {},
        "context_difference": "",
        "has_prev_context": False,
        "route_type": "",
        "current_plan": "",
        "plan_score": 0.0,
        "plan_feedback": "",
        "plan_iteration_count": 0,
        "plan_history": [],
        "clarification_questions": [],
        "suggestions": [],
        "user_responses": [],
        "user_approved": False,
        "user_feedback": "",
        "implementation_status": {},
        "files_modified": [],
        "implement_iteration_count": 0,
        "execution_log": [],
        "rag_query_results": [],
        "rag_fallback_used": False,
        "rag_fallback_results": [],
        "available_mcp_tools": [],
        "mcp_server_status": {},
        "active_provider": "",
        "active_model": "",
        "provider_switch_reason": "",
        "execution_mode": "confirm",
        "messages": [],
        "session_id": "",
        "time_started": "",
        "current_node": "",
        "error_log": [],
    }


async def _startup_mcp_discovery(config: dict) -> Dict[str, Any]:
    """
    Connect to MCP servers at startup and discover tool list, per design doc.
    """
    sys.path.insert(0, "CodePilot_UNCC")
    from mcp_client.client import MCPClient

    client = MCPClient(config)
    await client.connect_all()
    tools = client.get_tools()
    status = {
        name: ("connected" if name in client.sessions else "disconnected")
        for name in (config.get("mcp_servers") or {}).keys()
    }
    await client.cleanup()
    return {"available_mcp_tools": tools, "mcp_server_status": status}


def _display_node_event(node_name: str, node_output: dict) -> None:
    """Display node progress in the CLI."""
    console.print(f"   ", end="")

    if node_name == "super_router" and "route_type" in node_output:
        console.print(f"route → [bold]{node_output['route_type']}[/bold]")
    elif node_name == "plan_node" and "plan_score" in node_output:
        console.print(f"plan score: [bold]{node_output['plan_score']:.2f}[/bold]")
    elif node_name == "code_judge" and "implementation_status" in node_output:
        status = node_output["implementation_status"].get("status", "")
        color = "green" if status == "correct" else "red"
        console.print(f"verdict: [{color}]{status}[/{color}]")
    elif node_name == "implement" and "execution_log" in node_output:
        n_calls = len(node_output.get("execution_log", []))
        n_files = len(node_output.get("files_modified", []))
        console.print(f"tool calls: {n_calls}, files modified: {n_files}")
    else:
        console.print("[dim]done[/dim]")


def process_instruction(instruction: str, *, graph, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a user instruction using graph.stream() for progressive node output.
    In confirm mode, handles tool confirmation prompts from the background thread.
    """
    import threading
    import queue

    state["user_query"] = instruction
    state["user_approved"] = None
    state["route_type"] = ""

    cfg = {"configurable": {"thread_id": state.get("session_id") or "default-thread"}}
    final_state = state.copy()
    execution_mode = state.get("execution_mode", "auto")
    confirm = execution_mode.lower() in {"confirm", "confirmation"}

    # Import confirm queues from implement node
    from nodes.implement import _confirm_request_q, _confirm_response_q

    # Run graph.stream() in a background thread so main thread can handle confirmations
    events_q: queue.Queue = queue.Queue()
    error_holder: list = []

    def _run_graph():
        try:
            for event in graph.stream(state, cfg):
                events_q.put(("event", event))
            events_q.put(("done", None))
        except Exception as e:
            events_q.put(("error", e))

    graph_thread = threading.Thread(target=_run_graph, daemon=True)
    graph_thread.start()

    # Main loop: process graph events + handle confirmations
    while True:
        # Check for confirmation requests (non-blocking)
        if confirm:
            try:
                tool_name, args_preview = _confirm_request_q.get_nowait()
                console.print(
                    f"\n   [bold yellow]⚡ Tool call:[/bold yellow] "
                    f"[cyan]{tool_name}[/cyan]({args_preview})"
                )
                ans = input("   Approve? (y/n) ").strip().lower()
                _confirm_response_q.put(ans in {"y", "yes", ""})
                continue
            except queue.Empty:
                pass

        # Check for graph events
        try:
            msg_type, payload = events_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if msg_type == "done":
            break
        elif msg_type == "error":
            raise payload
        elif msg_type == "event":
            for node_name, node_output in payload.items():
                if node_name == "__end__":
                    continue
                _display_node_event(node_name, node_output)
                if isinstance(node_output, dict):
                    final_state.update(node_output)

    graph_thread.join(timeout=5)
    return final_state


def main(graph=None, config: dict | None = None):
    os.system("cls" if os.name == "nt" else "clear")
    print_welcome()

    session = PromptSession(
        history=InMemoryHistory(),
    )

    # Session state (persists across user messages)
    state: Dict[str, Any] = _default_state()
    if config is not None:
        state["execution_mode"] = config.get("execution_mode", "confirm")

        # Show which project directory the agent will work on
        project_path = config.get("project_path", os.getcwd())
        console.print(f"  [bold]Project:[/bold] [cyan]{project_path}[/cyan]\n")

        # MCP discovery at startup (writes available_mcp_tools + mcp_server_status)
        try:
            mcp_state = asyncio.run(_startup_mcp_discovery(config))
            state.update(mcp_state)
        except Exception:
            pass

    while True:
        try:
            console.rule(style="dim")
            user_input = session.prompt(
                HTML("<b><style fg='#00d4ff'>❯ </style></b>"),
            ).strip()
            console.rule(style="dim")
            print_status_bar()

            if not user_input:
                continue

            if user_input.startswith("/"):
                if user_input.lower().startswith("/mode"):
                    parts = user_input.split()
                    if len(parts) == 2 and parts[1].lower() in ("confirm", "auto"):
                        state["execution_mode"] = parts[1].lower()
                        console.print(f"[green]execution_mode set to[/green] {state['execution_mode']}")
                    else:
                        console.print("[yellow]Usage:[/yellow] /mode confirm | /mode auto")
                    continue

                if not handle_command(user_input):
                    break
                continue

            if graph is None:
                console.print("[red]Graph is not initialized.[/red]")
                continue

            state = process_instruction(user_input, graph=graph, state=state)
            console.print()

            impl = state.get("implementation_status") or {}
            route_type = state.get("route_type", "")
            plan_score = state.get("plan_score", 0.0)
            # Only show plan score for complex tasks that went through planning
            plan_line = f"[bold]Plan score:[/bold] {plan_score}\n" if route_type == "complex" else ""
            console.print(
                Panel(
                    f"[bold]Route:[/bold] {route_type or 'n/a'}\n"
                    f"{plan_line}"
                    f"[bold]Implementation status:[/bold] {impl.get('status','')}\n",
                    title="[bold]Run Summary[/bold]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
            break


if __name__ == "__main__":
    main()
