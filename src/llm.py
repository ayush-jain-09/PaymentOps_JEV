"""
LLM provider factory.

Supports:
  - openrouter   (default) — OpenRouter via OpenAI-compatible API

To add a new provider (e.g. Gemini, Anthropic):
  1. Add its config keys to src/config.py
  2. Add a branch in _build_llm() below
  3. Set LLM_PROVIDER=<new_provider> in .env

Nothing else in the codebase needs to change.
"""
import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    LLM_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from src.models import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PaymentOps Assistant, an AI agent for payment operations support.
You help support agents understand payment transactions, diagnose failures, and manage refunds.
You receive structured data from tools — summarize it clearly and professionally.
Never make up transaction data. If a tool returned an error, explain it politely.
Keep responses concise (2–4 sentences unless detail is needed)."""


def _build_llm():
    """Instantiate the LLM based on LLM_PROVIDER env variable."""
    provider = LLM_PROVIDER.lower().strip()

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Copy .env.example to .env and add your OpenRouter key."
            )
        from langchain_openai import ChatOpenAI

        logger.info("Initialising LLM via OpenRouter (model=%s)", OPENROUTER_MODEL)
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=0.2,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            # OpenRouter recommends these headers for attribution; they do not
            # expose any secret — they are sent as plain HTTP headers.
            default_headers={
                "HTTP-Referer": "https://github.com/paymentops-agent",
                "X-Title": "PaymentOps Agent",
            },
        )

    # ── Future providers ──────────────────────────────────────────────────────
    # elif provider == "openai":
    #     from src.config import OPENAI_API_KEY, OPENAI_MODEL
    #     from langchain_openai import ChatOpenAI
    #     return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    #
    # elif provider == "anthropic":
    #     from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0.2, api_key=ANTHROPIC_API_KEY)
    #
    # elif provider == "gemini":
    #     from src.config import GOOGLE_API_KEY, GEMINI_MODEL
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2, google_api_key=GOOGLE_API_KEY)

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
            "Supported values: openrouter"
        )


@lru_cache(maxsize=1)
def get_llm():
    """Return the configured LLM singleton (loaded once per process)."""
    return _build_llm()


def get_llm_response(state: AgentState) -> str:
    """Generate a natural-language response using the configured LLM."""
    try:
        llm = get_llm()
    except ValueError as e:
        return f"⚠️ LLM unavailable: {e}"
    except Exception as e:
        return f"⚠️ LLM initialization failed: {e}"

    tool_name = state.get("tool_name", "none")
    tool_result = state.get("tool_result", {})
    gate_decision = state.get("gate_decision", "")
    query = state.get("query", "")
    routing_confidence = state.get("routing_confidence", 0.0)

    context_parts = [f"User query: {query}"]

    if tool_name == "low_confidence":
        context_parts.append(
            f"Routing confidence was too low ({routing_confidence:.0%}). "
            "Ask the user to clarify their request."
        )
    elif tool_name == "none":
        context_parts.append("No specific payment tool was needed for this query.")
    elif gate_decision == "blocked":
        context_parts.append(
            "A refund was requested but was BLOCKED by the action gate policy. "
            "Explain this clearly and suggest the user contact their account manager."
        )
    elif gate_decision == "requires_approval":
        tx_id = tool_result.get("tx_id", "the transaction") if tool_result else "the transaction"
        context_parts.append(
            f"A refund was requested for {tx_id} but REQUIRES HUMAN APPROVAL before it can be processed. "
            "Explain this and tell the user to await confirmation."
        )
    elif tool_result:
        if "error" in tool_result:
            context_parts.append(f"Tool error: {tool_result['error']}")
        else:
            context_parts.append(f"Tool result ({tool_name}): {tool_result}")

    human_prompt = "\n".join(context_parts)

    try:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_prompt)]
        )
        return response.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"⚠️ LLM call failed: {e}"
