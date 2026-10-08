import pytest
import src.laya_router as laya_router_module
from src.graph import node_laya_tool_router


def _make_predict(qid, choice, confidence):
    probs = {choice: confidence}
    other = max(0.0, (1.0 - confidence) / 4)
    for lbl in ["transaction_lookup", "incident_search", "calculator", "refund_transaction", "none"]:
        if lbl != choice:
            probs[lbl] = other
    return {
        "answers": {
            qid: {
                "choice": choice,
                "answer_confidence": confidence,
                "probabilities": probs,
            }
        },
        "routing": {"model": "mock"},
    }


def _patch_router(monkeypatch, choice, confidence):
    mock_result = _make_predict("tool", choice, confidence)
    monkeypatch.setattr(laya_router_module, "LAYA_AVAILABLE", True)
    monkeypatch.setattr(laya_router_module.router, "predict", lambda state, questions: mock_result)


def test_routes_transaction_lookup(monkeypatch):
    _patch_router(monkeypatch, "transaction_lookup", 0.95)
    result = node_laya_tool_router({"query": "What amount was charged for TX-1042?", "trace": []})
    assert result["tool_name"] == "transaction_lookup"
    assert result["routing_confidence"] == pytest.approx(0.95)


def test_routes_incident_search(monkeypatch):
    _patch_router(monkeypatch, "incident_search", 0.88)
    result = node_laya_tool_router({"query": "Why did TX-1003 fail?", "trace": []})
    assert result["tool_name"] == "incident_search"


def test_routes_calculator(monkeypatch):
    _patch_router(monkeypatch, "calculator", 0.82)
    result = node_laya_tool_router({"query": "Calculate 80% of 4999", "trace": []})
    assert result["tool_name"] == "calculator"


def test_routes_refund(monkeypatch):
    _patch_router(monkeypatch, "refund_transaction", 0.91)
    result = node_laya_tool_router({"query": "Refund TX-1042", "trace": []})
    assert result["tool_name"] == "refund_transaction"


def test_routes_none(monkeypatch):
    _patch_router(monkeypatch, "none", 0.85)
    result = node_laya_tool_router({"query": "Hello", "trace": []})
    assert result["tool_name"] == "none"


def test_low_confidence_fallback(monkeypatch):
    _patch_router(monkeypatch, "transaction_lookup", 0.45)
    result = node_laya_tool_router({"query": "Some ambiguous query", "trace": []})
    assert result["tool_name"] == "low_confidence"
