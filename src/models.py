from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    query: str
    tool_name: str
    tool_result: dict
    routing_decision: str
    routing_confidence: float
    routing_probabilities: dict
    gate_decision: str
    gate_confidence: float
    gate_probabilities: dict
    llm_response: str
    trace: list
    error: str | None
