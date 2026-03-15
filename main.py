#!/usr/bin/env python3
"""
CodePilot — AI-Powered CLI Coding Agent
Entry point: initializes config, builds graph, starts CLI.
"""

from core.config import load_config
from core.graph import build_graph


def main():
    config = load_config()
    graph = build_graph(config)
    from ui.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
