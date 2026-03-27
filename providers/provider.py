"""
Provider Abstraction
Uses LangChain's BaseChatModel to support Ollama, Groq, and OpenAI.

Priority for cloud provider (used for complex tasks, implement, code_judge):
  1. OpenAI  — if OPENAI_API_KEY is set
  2. Groq    — if GROQ_API_KEY is set
  3. Ollama  — always available (local fallback)

Routing logic:
  - No cloud key at all        → always Ollama
  - Cloud key available:
      - simple tasks / pre-routing → Ollama  (fast, local)
      - complex / force_cloud      → best available cloud provider
"""

import os
import logging

logger = logging.getLogger(__name__)

OLLAMA_DEFAULTS = {"name": "ollama", "model": "llama3.1:8b"}
GROQ_DEFAULTS = {"name": "groq", "model": "llama-3.3-70b-versatile"}
OPENAI_DEFAULTS = {"name": "openai", "model": "gpt-4o-mini"}


def _has_key(env_var: str) -> bool:
    return bool(os.environ.get(env_var))


def _build_llm(name: str, model: str, config: dict):
    """Instantiate a LangChain chat model by provider name."""
    if name == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "")
        return ChatGroq(model=model, api_key=api_key)
    elif name == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    elif name == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return ChatOpenAI(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {name}")


def _get_cloud_provider(config: dict) -> tuple[str, str] | None:
    """
    Return (name, model) for the best available cloud provider.
    Priority: OpenAI > Groq.
    Returns None if no cloud key is available.
    """
    provider_cfg = config.get("provider") or {}

    # Check OpenAI first
    if _has_key("OPENAI_API_KEY"):
        # If config explicitly sets openai, use its model; otherwise use default
        if provider_cfg.get("name") == "openai":
            return "openai", provider_cfg.get("model", OPENAI_DEFAULTS["model"])
        return OPENAI_DEFAULTS["name"], OPENAI_DEFAULTS["model"]

    # Then Groq
    if _has_key("GROQ_API_KEY"):
        if provider_cfg.get("name") == "groq":
            return "groq", provider_cfg.get("model", GROQ_DEFAULTS["model"])
        return GROQ_DEFAULTS["name"], GROQ_DEFAULTS["model"]

    return None


def get_llm(config: dict, route_type: str = None, force_cloud: bool = False):
    """
    Return a LangChain chat model based on config and task complexity.

    Args:
        config: The loaded config dict.
        route_type: "simple" or "complex" (from super_router).
        force_cloud: If True, use the best cloud provider regardless of
                     route_type. Used for nodes that need strong
                     tool-calling ability (e.g. implement, code_judge).
    """
    cloud = _get_cloud_provider(config)

    if cloud is None:
        # No cloud key — always use Ollama
        logger.info("No cloud API key found, defaulting to Ollama (%s)", OLLAMA_DEFAULTS["model"])
        return _build_llm(OLLAMA_DEFAULTS["name"], OLLAMA_DEFAULTS["model"], config)

    cloud_name, cloud_model = cloud

    # force_cloud: always use best cloud provider (implement, code_judge)
    if force_cloud:
        logger.info("force_cloud → using %s (%s)", cloud_name, cloud_model)
        return _build_llm(cloud_name, cloud_model, config)

    # Route based on complexity
    if route_type == "simple":
        logger.info("Simple task → using Ollama (%s)", OLLAMA_DEFAULTS["model"])
        return _build_llm(OLLAMA_DEFAULTS["name"], OLLAMA_DEFAULTS["model"], config)
    elif route_type == "complex":
        logger.info("Complex task → using %s (%s)", cloud_name, cloud_model)
        return _build_llm(cloud_name, cloud_model, config)
    else:
        # Pre-routing → Ollama
        logger.info("Pre-routing → using Ollama (%s)", OLLAMA_DEFAULTS["model"])
        return _build_llm(OLLAMA_DEFAULTS["name"], OLLAMA_DEFAULTS["model"], config)
