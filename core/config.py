"""
Configuration loader — reads config.yaml and returns a config dict.
Resolves project_path and internal paths relative to CodePilot_UNCC root.
"""

import os
import sys
import yaml

# CodePilot_UNCC root directory (parent of core/)
AGENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(AGENT_ROOT, "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Store agent root so other modules can find internal files
    config["agent_root"] = AGENT_ROOT

    # Resolve project_path to absolute (relative to AGENT_ROOT, not cwd)
    raw_path = config.get("project_path", ".")
    if os.path.isabs(raw_path):
        config["project_path"] = raw_path
    else:
        config["project_path"] = os.path.normpath(os.path.join(AGENT_ROOT, raw_path))

    # Resolve checkpointing path relative to agent root
    cp = config.get("checkpointing") or {}
    if cp.get("path"):
        cp_path = cp["path"]
        if not os.path.isabs(cp_path):
            cp["path"] = os.path.join(AGENT_ROOT, ".codepilot", "checkpoints.pkl")

    # Resolve MCP server paths
    mcp_servers = config.get("mcp_servers") or {}

    # RAG server: resolve script path relative to agent root
    rag = mcp_servers.get("rag") or {}
    if rag.get("args"):
        rag["args"] = [
            os.path.join(AGENT_ROOT, arg) if not os.path.isabs(arg) and arg.startswith("mcp_servers/") else arg
            for arg in rag["args"]
        ]
    # Use the current Python interpreter for the RAG server (same venv)
    if rag.get("command") and "venv" in rag["command"]:
        rag["command"] = sys.executable

    return config
