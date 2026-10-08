import logging
from src.config import LAYA_DEVICE

logger = logging.getLogger(__name__)

LAYA_AVAILABLE = False
router = None

try:
    from laya import Router
    # preload=False: the Router object is created immediately without downloading
    # the model weights. The model is lazily loaded on the first predict() call.
    # This keeps import-time fast and lets tests monkeypatch `router.predict`
    # before any real inference happens.
    router = Router(preload=False, device=LAYA_DEVICE)
    LAYA_AVAILABLE = True
    logger.info("Laya Router created (lazy load) on device=%s", LAYA_DEVICE)
except Exception as exc:
    logger.error("Failed to load Laya Router: %s", exc)
    router = None

TOOL_ROUTING_QUESTIONS = {
    "tool": {
        "type": "choice",
        "instructions": "Which payment operations tool should handle this request?",
        "criteria": {
            "transaction_lookup": "look up, retrieve, or get details about a specific transaction by ID",
            "incident_search": "search for known payment incidents, error explanations, runbooks, or troubleshooting guides",
            "calculator": "calculate a value such as refund amount, fee percentage, or settlement difference",
            "refund_transaction": "initiate, process, or execute a refund for a transaction",
            "none": "general questions, greetings, or requests that do not match any payment tool",
        }
    }
}

ACTION_GATE_QUESTIONS = {
    "gate": {
        "type": "choice",
        "instructions": "Should this refund action be allowed to proceed?",
        "criteria": {
            "safe": "a straightforward low-risk refund request with a clear valid transaction ID",
            "requires_approval": "a refund request that needs human review or supervisor approval before proceeding",
            "blocked": "a suspicious, unauthorized, or explicitly disallowed refund attempt",
        }
    }
}


def route_to_tool(query: str) -> dict:
    """Ask Laya which tool to use for the given query."""
    if not LAYA_AVAILABLE or router is None:
        return {"tool": "none", "confidence": 0.0, "probabilities": {}, "error": "Laya not available"}
    try:
        state = {"query": query}
        result = router.predict(state, TOOL_ROUTING_QUESTIONS)
        ans = result["answers"]["tool"]
        return {
            "tool": ans["choice"],
            "confidence": ans["answer_confidence"],
            "probabilities": ans.get("probabilities", {}),
            "error": None,
        }
    except Exception as exc:
        logger.error("Laya routing failed: %s", exc)
        return {"tool": "none", "confidence": 0.0, "probabilities": {}, "error": str(exc)}


def gate_action(query: str, context: dict | None = None) -> dict:
    """Ask Laya whether a write/refund action should proceed."""
    if not LAYA_AVAILABLE or router is None:
        return {"decision": "blocked", "confidence": 0.0, "probabilities": {}, "error": "Laya not available"}
    try:
        state = {"query": query}
        if context:
            state.update(context)
        result = router.predict(state, ACTION_GATE_QUESTIONS)
        ans = result["answers"]["gate"]
        return {
            "decision": ans["choice"],
            "confidence": ans["answer_confidence"],
            "probabilities": ans.get("probabilities", {}),
            "error": None,
        }
    except Exception as exc:
        logger.error("Laya action gate failed: %s", exc)
        return {"decision": "blocked", "confidence": 0.0, "probabilities": {}, "error": str(exc)}
