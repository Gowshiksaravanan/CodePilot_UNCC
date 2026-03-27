#!/usr/bin/env python3
"""
CodePilot — AI-Powered CLI Coding Agent
Entry point: initializes config, builds graph, starts CLI.

Usage:
    python main.py                     # uses project_path from config.yaml (default: ".")
    python main.py /path/to/project    # override project_path via CLI arg
"""

import os
import sys

from dotenv import load_dotenv
# Load .env from the agent root (CodePilot_UNCC/) regardless of cwd
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from core.config import load_config
from core.graph import build_graph


def main():
    config = load_config()

    # Allow CLI override: python main.py /path/to/project
    if len(sys.argv) > 1:
        project_path = os.path.abspath(sys.argv[1])
        if not os.path.isdir(project_path):
            print(f"Error: '{project_path}' is not a valid directory.")
            sys.exit(1)
        config["project_path"] = project_path

    graph = build_graph(config)
    from ui.cli import main as cli_main
    cli_main(graph=graph, config=config)


if __name__ == "__main__":
    main()
