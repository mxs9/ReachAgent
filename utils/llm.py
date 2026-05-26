"""
Multi-provider LLM client — OpenAI / DeepSeek / Anthropic.

Provider auto-detected from env (priority order):
  1. LLM_PROVIDER=openai|deepseek|anthropic  (explicit override)
  2. DEEPSEEK_API_KEY  → deepseek
  3. OPENAI_API_KEY    → openai
  4. ANTHROPIC_API_KEY → anthropic
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()


def _detect() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("openai", "deepseek", "anthropic"):
        return explicit
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No LLM key found. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env"
    )


PROVIDER = _detect()

_MODEL = {
    "openai":    os.getenv("OPENAI_MODEL",    "gpt-4o-mini"),
    "deepseek":  os.getenv("DEEPSEEK_MODEL",  "deepseek-chat"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
}


def _oai_client():
    from openai import OpenAI
    if PROVIDER == "deepseek":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str) -> str:
    """Plain-text LLM call."""
    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        r = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=_MODEL["anthropic"],
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    r = _oai_client().chat.completions.create(
        model=_MODEL[PROVIDER],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return r.choices[0].message.content.strip()


def call_llm_structured(prompt: str, schema: dict, tool_name: str) -> dict:
    """Structured output via tool call — returns a parsed dict, never raw text."""
    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        r = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=_MODEL["anthropic"],
            max_tokens=2048,
            tools=[{"name": tool_name, "description": tool_name, "input_schema": schema}],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in r.content:
            if block.type == "tool_use":
                return block.input
        return {}

    tool = {"type": "function", "function": {"name": tool_name, "parameters": schema}}
    r = _oai_client().chat.completions.create(
        model=_MODEL[PROVIDER],
        messages=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=0.3,
    )
    return json.loads(r.choices[0].message.tool_calls[0].function.arguments)
