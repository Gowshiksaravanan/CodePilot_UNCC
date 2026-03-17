"""
Provider Abstraction
Uses LangChain's BaseChatModel to support Ollama, Groq, and OpenAI.
Configured via config.yaml.
"""

import os


def get_llm(config: dict):
    """Return a LangChain chat model instance based on config."""
    provider = config["provider"]
    name = provider["name"]
    model = provider["model"]

    if name == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=os.environ.get(provider.get("api_key_env", "GROQ_API_KEY"), ""),
        )
    elif name == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model)
    elif name == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get(provider.get("api_key_env", "OPENAI_API_KEY"), ""),
        )
    else:
        raise ValueError(f"Unknown provider: {name}")
