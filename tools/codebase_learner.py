"""
Codebase Learner Tool
Scans the project directory to gather file tree, tech stack, configs, and dependencies.
Used by: context_updator
"""

import os
from pathlib import Path


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".sh",
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    ".idea", ".vscode", "coverage", "htmlcov",
}

MAX_FILE_SIZE = 50_000
MAX_FILES = 300


def _language_from_extension(ext: str) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c/cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".txt": "text",
        ".sh": "shell",
    }
    return mapping.get(ext, "other")


def scan_codebase(project_path: str) -> dict:
    """Scan a project and return a lightweight codebase snapshot."""
    root = project_path or "."
    files = {}
    languages = {}
    total_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for filename in filenames:
            if total_files >= MAX_FILES:
                break

            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root)

            ext = Path(filename).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            try:
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    continue

                mtime = os.path.getmtime(full_path)
                language = _language_from_extension(ext)

                files[rel_path] = {
                    "size": size,
                    "mtime": mtime,
                    "language": language,
                }
                languages[language] = languages.get(language, 0) + 1
                total_files += 1
            except OSError:
                continue

    return {
        "files": files,
        "languages": languages,
        "total_files": total_files,
        "project_path": os.path.abspath(root),
    }
