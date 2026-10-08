import logging
from functools import lru_cache
from src.config import LLM_MODEL, OPENAI_API_KEY
from src.models import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PaymentOps Assistant, an AI agent for payment operations support.
You help support agents understand payment transactions, diagnose failures, and manage refunds.
You receive structured data from tools — summarize it clearly and professionally.
Never make up transaction data. If a tool returned an error, explain it politely.
Keep responses concise (2–4 sentences unless detail is needed)."""


@lru_cache(maxsize=1)
def get_llm():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    except ImportError as e:
        raise ImportError(f"langchain_openai not installed: {e}")


def get_llm_response(state: AgentState) -> str:
    """Generate a natural language response using the LLM."""
    try:
        llm = get_llm()
    except ValueError as e:
        return f"\u26a0\ufe0f LLM unavailable: {e}"
    except Exception as e:
        return f"\u26a0\ufe0f LLM initialization failed: {e}"

    tool_name = state.get("tool_name", "none")
    tool_result = state.get("tool_result", {})
    gate_decision = state.get("gate_decision", "")
    query = state.get("query", "")
    routing_confidence = state.get("routing_confidence", 0.0)

    context_parts = [f"User query: {query}"]

    if tool_name == "low_confidence":
        context_parts.append(
            f"Routing confidence was too low ({routing_confidence:.0%}). Ask the user to clarify their request."
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
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_prompt)])
        return response.content
    except Exception as e:
        return f"\u26a0\ufe0f LLM call failed: {e}"
