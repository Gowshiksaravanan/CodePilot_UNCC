"""
Context Updator Node
====================
Reads:  (project directory on disk + comparator output)
Writes: code_base, context, has_prev_context
"""

import logging
import os
from pathlib import Path

from core.state import AgentState

logger = logging.getLogger(__name__)

# File extensions to read as code
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".sh", ".env.example",
}

# Directories to always skip
IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    ".idea", ".vscode", "coverage", "htmlcov",
}

# Max file size to read (bytes) — skip large files
MAX_FILE_SIZE = 50_000

# Max total files to read — avoid overwhelming the context
MAX_FILES = 100


def _walk_project(root: str) -> dict[str, str]:
    """
    Walk the project directory and return {relative_path: content}
    for all relevant code files.
    """
    code_base = {}
    files_read = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for filename in filenames:
            if files_read >= MAX_FILES:
                logger.warning("context_updator: reached MAX_FILES=%d limit", MAX_FILES)
                return code_base

            ext = Path(filename).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path  = os.path.relpath(full_path, root)

            try:
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    logger.debug("Skipping large file: %s (%d bytes)", rel_path, size)
                    continue

                content = Path(full_path).read_text(encoding="utf-8", errors="ignore")
                code_base[rel_path] = content
                files_read += 1

            except Exception:
                logger.exception("Failed to read file: %s", rel_path)

    return code_base


def _build_context_summary(code_base: dict[str, str]) -> dict:
    """Build a lightweight context summary from the scanned code_base."""
    file_list = sorted(code_base.keys())

    # Group by directory
    dirs: dict[str, list[str]] = {}
    for path in file_list:
        folder = str(Path(path).parent)
        dirs.setdefault(folder, []).append(Path(path).name)

    return {
        "total_files":  len(file_list),
        "directories":  dirs,
        "file_list":    file_list,
    }


def run(state: AgentState) -> dict:
    """
    Walks the current project directory, reads all relevant code files,
    and updates code_base + context in state.
    """
    root = os.getcwd()
    logger.info("context_updator: scanning project at %s", root)

    existing_code_base = state.get("code_base") or {}

    # Scan the project
    code_base = _walk_project(root)

    if not code_base:
        logger.warning("context_updator: no code files found in %s", root)

    # Determine if there was previous context
    has_prev_context = bool(existing_code_base)

    # Build summary context
    context = _build_context_summary(code_base)

    logger.info(
        "context_updator: loaded %d files from %d directories",
        context["total_files"],
        len(context["directories"]),
    )

    return {
        "code_base":       code_base,
        "context":         context,
        "has_prev_context": has_prev_context,
        "current_node":    "context_updator",
    }