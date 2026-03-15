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

    console.print(f"[yellow]Unknown command:[/yellow] {command}")
    return True


def process_instruction(instruction: str):
    """
    Process a user instruction.
    ────────────────────────────
    Hook point — replace the placeholder below with actual agent logic.
    """
    console.print()
    console.print(
        Panel(
            f"[dim]Received instruction:[/dim]\n\n\"{instruction}\"\n\n"
            "[yellow]Agent processing not yet implemented.\n"
            "This is where the LangGraph pipeline will be invoked.[/yellow]",
            title="[bold]CodePilot Response[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print_welcome()

    session = PromptSession(
        history=InMemoryHistory(),
    )

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
                if not handle_command(user_input):
                    break
                continue

            process_instruction(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
            break


if __name__ == "__main__":
    main()
