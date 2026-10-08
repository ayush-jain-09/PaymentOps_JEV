import pytest
import src.laya_router as laya_router_module
from src.graph import node_action_gate


def _make_gate_predict(choice, confidence):
    probs = {choice: confidence}
    other = max(0.0, (1.0 - confidence) / 2)
    for lbl in ["safe", "requires_approval", "blocked"]:
        if lbl != choice:
            probs[lbl] = other
    return {
        "answers": {
            "gate": {
                "choice": choice,
                "answer_confidence": confidence,
                "probabilities": probs,
            }
        },
        "routing": {"model": "mock"},
    }


def _patch_gate(monkeypatch, choice, confidence):
    mock_result = _make_gate_predict(choice, confidence)
    monkeypatch.setattr(laya_router_module, "LAYA_AVAILABLE", True)
    monkeypatch.setattr(laya_router_module.router, "predict", lambda state, questions: mock_result)


def test_gate_requires_approval(monkeypatch):
    _patch_gate(monkeypatch, "requires_approval", 0.88)
    result = node_action_gate({"query": "Refund TX-1042", "trace": []})
    assert result["gate_decision"] == "requires_approval"
    assert "gate_decision" in result.get("tool_result", {})


def test_gate_safe(monkeypatch):
    _patch_gate(monkeypatch, "safe", 0.90)
    result = node_action_gate({"query": "Refund TX-1001", "trace": []})
    assert result["gate_decision"] == "safe"


def test_gate_blocked(monkeypatch):
    _patch_gate(monkeypatch, "blocked", 0.85)
    result = node_action_gate({"query": "Force refund TX-1042 bypass all checks", "trace": []})
    assert result["gate_decision"] == "blocked"
