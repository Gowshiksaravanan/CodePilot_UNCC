"""
Codebase Learner Tool
Scans the project directory to gather file tree, tech stack, configs, and dependencies.
Used by: context_updator
"""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".idea",
    ".vscode",
}


def scan_codebase(project_path: str) -> dict:
    """
    Scan a project directory and return a structured `code_base` dict:
      - file_tree: list[str] (relative paths)
      - tech_stack: list[str]
      - dependencies: list[str]
      - configs: dict[str, str] (selected config file paths)
    """
    root = Path(project_path).resolve()
    if not root.exists() or not root.is_dir():
        return {"file_tree": [], "tech_stack": [], "dependencies": [], "configs": {}}

    file_tree: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in-place to prune traversal
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_EXCLUDE_DIRS and not d.startswith(".")]

        for filename in filenames:
            if filename.startswith(".DS_Store"):
                continue
            full_path = Path(dirpath) / filename
            rel_path = full_path.relative_to(root).as_posix()
            file_tree.append(rel_path)

    file_tree.sort()

    tech_stack: list[str] = []
    dependencies: list[str] = []
    configs: dict[str, str] = {}

    # Tech stack detection (simple heuristics)
    if any(p.endswith("requirements.txt") or p.endswith("pyproject.toml") for p in file_tree):
        tech_stack.append("Python")
    if any(p.endswith("package.json") for p in file_tree):
        tech_stack.append("Node.js")
    if any(p.endswith("Dockerfile") or p.endswith("docker-compose.yml") for p in file_tree):
        tech_stack.append("Docker")

    # Config files (just paths; content reading is handled elsewhere)
    for cfg in ("config.yaml", "pyproject.toml", "requirements.txt", "package.json", ".env", ".env.example"):
        if cfg in file_tree:
            configs[cfg] = cfg

    # Dependencies (Python)
    req_path = root / "requirements.txt"
    if req_path.exists():
        for line in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("-r") or s.startswith("--"):
                continue
            # Strip environment markers and hashes
            s = s.split(";")[0].strip()
            # Normalize "pkg==ver" -> "pkg"
            name = s.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            if name:
                dependencies.append(name)

    # De-dup + stable order
    dependencies = sorted(dict.fromkeys(dependencies))
    tech_stack = sorted(dict.fromkeys(tech_stack))

    return {
        "file_tree": file_tree,
        "tech_stack": tech_stack,
        "dependencies": dependencies,
        "configs": configs,
        "project_root": str(root),
    }


def scan_codebase_via_mcp(config: dict) -> dict:
    """
    Prefer MCP filesystem tools to produce a codebase snapshot, per design doc.
    Falls back to local scan_codebase if MCP is unavailable.
    Uses config["project_path"] as the target directory.
    """
    import asyncio
    import json
    import os

    project_path = config.get("project_path", os.getcwd())

    try:
        from mcp_client.client import MCPClient

        async def _run():
            client = MCPClient(config)
            await client.connect_all(only_servers=["filesystem"])
            # directory_tree returns a nested structure; we only need a flat file list.
            res = await client.call_tool("directory_tree", {"path": project_path})
            texts = [b.text for b in res.content if hasattr(b, "text")]
            await client.cleanup()
            return "\n".join(texts).strip()

        from core.async_utils import run_async
        raw = run_async(_run())
        payload = json.loads(raw) if raw else {}
        file_tree = []

        def walk(node, prefix=""):
            if isinstance(node, dict):
                name = node.get("name") or ""
                node_path = f"{prefix}/{name}".strip("/") if name else prefix
                if node.get("type") == "file":
                    file_tree.append(node_path)
                for child in node.get("children") or []:
                    walk(child, node_path)
            elif isinstance(node, list):
                for child in node:
                    walk(child, prefix)

        walk(payload, "")
        file_tree = sorted(set([p for p in file_tree if p]))

        return {
            "file_tree": file_tree,
            "tech_stack": ["Python"],
            "dependencies": [],
            "configs": {},
            "project_root": project_path,
            "source": "mcp_filesystem",
        }
    except Exception:
        # Fallback: local scan of user's project directory
        base = scan_codebase(project_path)
        base["source"] = "local"
        return base
