import pytest
from src.models import AgentState


def make_predict_result(qid: str, choice: str, confidence: float) -> dict:
    """Build a properly shaped laya Router.predict() result."""
    labels = [choice, "other"]
    other_prob = max(0.0, (1.0 - confidence) / max(len(labels) - 1, 1))
    probs = {choice: confidence}
    # fill remaining probability
    for lbl in ["transaction_lookup", "incident_search", "calculator", "refund_transaction", "none",
                "safe", "requires_approval", "blocked"]:
        if lbl != choice:
            probs[lbl] = other_prob
    return {
        "answers": {
            qid: {
                "choice": choice,
                "answer_confidence": confidence,
                "probabilities": probs,
            }
        },
        "routing": {"model": "mock-laya-checkpoint"},
    }


@pytest.fixture
def mock_routing_result():
    def _make(tool: str, confidence: float) -> dict:
        return make_predict_result("tool", tool, confidence)
    return _make


@pytest.fixture
def mock_gate_result():
    def _make(decision: str, confidence: float) -> dict:
        return make_predict_result("gate", decision, confidence)
    return _make


@pytest.fixture
def sample_agent_state() -> AgentState:
    return AgentState(
        query="What amount was charged for TX-1042?",
        tool_name="transaction_lookup",
        tool_result={"id": "TX-1042", "amount": 4999.0, "currency": "INR", "status": "failed"},
        routing_decision="transaction_lookup",
        routing_confidence=0.91,
        routing_probabilities={"transaction_lookup": 0.91, "none": 0.09},
        gate_decision="",
        gate_confidence=0.0,
        gate_probabilities={},
        llm_response="",
        trace=[],
        error=None,
    )
