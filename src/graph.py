import re
import logging
from langgraph.graph import StateGraph, END
from src.models import AgentState
from src.laya_router import route_to_tool, gate_action
from src.tools import run_tool
from src.llm import get_llm_response
import src.config as config

logger = logging.getLogger(__name__)


def _extract_tx_id(query: str) -> str:
    m = re.search(r'TX-\d+', query, re.IGNORECASE)
    return m.group(0).upper() if m else "unknown"


def node_laya_tool_router(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = list(state.get("trace", []))

    routing = route_to_tool(query)
    tool_name = routing["tool"]
    confidence = routing["confidence"]
    probabilities = routing["probabilities"]

    trace.append({
        "step": "LAYA \u2014 TOOL ROUTER",
        "decision": tool_name,
        "confidence": confidence,
        "probabilities": probabilities,
        "error": routing.get("error"),
    })

    if confidence < config.ROUTING_CONFIDENCE_THRESHOLD:
        trace.append({
            "step": "SYSTEM",
            "message": f"Routing confidence {confidence:.0%} is below threshold {config.ROUTING_CONFIDENCE_THRESHOLD:.0%}. Falling back to clarification.",
        })
        tool_name = "low_confidence"

    return {
        "tool_name": tool_name,
        "routing_decision": routing["tool"],
        "routing_confidence": confidence,
        "routing_probabilities": probabilities,
        "trace": trace,
    }


def node_action_gate(state: AgentState) -> dict:
    query = state.get("query", "")
    trace = list(state.get("trace", []))

    gate = gate_action(query)
    decision = gate["decision"]
    confidence = gate["confidence"]

    trace.append({
        "step": "LAYA \u2014 ACTION GATE",
        "decision": decision,
        "confidence": confidence,
        "probabilities": gate["probabilities"],
        "error": gate.get("error"),
    })

    if decision != "safe":
        trace.append({
            "step": "SYSTEM",
            "message": f"Refund NOT executed. Gate decision: {decision}.",
        })

    return {
        "gate_decision": decision,
        "gate_confidence": confidence,
        "gate_probabilities": gate["probabilities"],
        "tool_result": {"tx_id": _extract_tx_id(query), "gate_decision": decision},
        "trace": trace,
    }


def node_tool_execution(state: AgentState) -> dict:
    tool_name = state.get("tool_name", "none")
    query = state.get("query", "")
    trace = list(state.get("trace", []))

    result = run_tool(tool_name, query)

    trace.append({
        "step": "TOOL EXECUTION",
        "tool": tool_name,
        "result": result,
    })

    return {"tool_result": result, "trace": trace}


def node_llm_response(state: AgentState) -> dict:
    trace = list(state.get("trace", []))
    response = get_llm_response(state)
    trace.append({
        "step": "LLM",
        "response": response,
    })
    return {"llm_response": response, "trace": trace}


def _route_after_router(state: AgentState) -> str:
    tool = state.get("tool_name", "none")
    if tool == "refund_transaction":
        return "action_gate"
    return "tool_execution"


def _route_after_gate(state: AgentState) -> str:
    decision = state.get("gate_decision", "blocked")
    if decision == "safe":
        return "tool_execution"
    return "llm_response"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("laya_tool_router", node_laya_tool_router)
    workflow.add_node("action_gate", node_action_gate)
    workflow.add_node("tool_execution", node_tool_execution)
    workflow.add_node("llm_response", node_llm_response)

    workflow.set_entry_point("laya_tool_router")
    workflow.add_conditional_edges("laya_tool_router", _route_after_router, {
        "action_gate": "action_gate",
        "tool_execution": "tool_execution",
    })
    workflow.add_conditional_edges("action_gate", _route_after_gate, {
        "tool_execution": "tool_execution",
        "llm_response": "llm_response",
    })
    workflow.add_edge("tool_execution", "llm_response")
    workflow.add_edge("llm_response", END)

    return workflow.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str) -> AgentState:
    """Run the agent graph for a given query. Returns the final state."""
    initial_state: AgentState = {
        "query": query,
        "trace": [{"step": "USER", "query": query}],
        "error": None,
    }
    graph = get_graph()
    result = graph.invoke(initial_state)
    return result
