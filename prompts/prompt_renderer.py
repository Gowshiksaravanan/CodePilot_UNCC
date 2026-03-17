"""
POML Prompt Renderer Module

Renders POML prompt templates with dynamic context variables.
Each tool has a .poml file in the prompts/ directory.
The renderer loads the file, injects context variables, renders via poml(),
and returns the prompt string.

Usage:
    from prompts.prompt_renderer import render_prompt

    content = render_prompt("plan_generator", {
        "restructured_query": "Add JWT auth",
        "code_base": "...",
        "context": "...",
    })
"""

import json
import logging
import os
from typing import Any, Dict

from poml import poml

logger = logging.getLogger(__name__)

# Directory where .poml files live
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache: cache_key -> rendered_content
_RENDER_CACHE: Dict[str, str] = {}


def get_poml_path(prompt_name: str) -> str:
    """Get the full path to a .poml file by name."""
    return os.path.join(PROMPTS_DIR, f"{prompt_name}.poml")


def render_prompt(prompt_name: str, context: Dict[str, Any]) -> str:
    """
    Render a POML prompt template with the given context variables.

    Args:
        prompt_name: Name of the .poml file (without extension).
                     e.g., "plan_generator", "plan_verifier"
        context: Dictionary of variables to inject into the template.
                 Keys must match {{placeholder}} names in the .poml file.

    Returns:
        Rendered prompt content as string.

    Raises:
        ValueError: If the .poml file cannot be found or rendered.
    """
    poml_path = get_poml_path(prompt_name)

    if not os.path.exists(poml_path):
        raise ValueError(f"POML template not found: {poml_path}")

    try:
        # Read the template
        with open(poml_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Replace {{placeholders}} with context values before POML rendering
        for key, value in context.items():
            placeholder = "{{" + key + "}}"
            str_value = str(value) if value else ""
            template_content = template_content.replace(placeholder, str_value)

        # Render using poml() with inline string
        result_raw = poml(template_content, chat=True, format="raw")

        # Parse the raw JSON result
        result = json.loads(result_raw)

        # Extract content from messages
        messages = result.get("messages", [])
        if isinstance(messages, list) and messages:
            content = messages[0].get("content", "")
        elif isinstance(messages, str):
            content = messages
        else:
            content = str(result)

        logger.debug("Rendered prompt '%s' (%d chars)", prompt_name, len(content))
        return content

    except FileNotFoundError:
        error_msg = f"POML template not found: {poml_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Failed to render prompt '{prompt_name}': {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e


def render_prompt_cached(prompt_name: str, context: Dict[str, Any]) -> str:
    """
    Render with caching. Uses a cache key based on prompt name + context hash.
    """
    cache_key = f"{prompt_name}:{hash(frozenset((k, str(v)) for k, v in sorted(context.items())))}"

    if cache_key in _RENDER_CACHE:
        logger.debug("Cache HIT for prompt '%s'", prompt_name)
        return _RENDER_CACHE[cache_key]

    content = render_prompt(prompt_name, context)
    _RENDER_CACHE[cache_key] = content
    logger.debug("Cached prompt '%s'", prompt_name)
    return content


def clear_cache() -> None:
    """Clear the rendered prompt cache."""
    _RENDER_CACHE.clear()
    logger.info("Prompt render cache cleared")
