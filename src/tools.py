import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Load data at module level
try:
    with open(DATA_DIR / "transactions.json", encoding="utf-8") as f:
        _tx_list = json.load(f)
    TRANSACTIONS: dict[str, dict] = {tx["id"].upper(): tx for tx in _tx_list}
except Exception as e:
    logger.error("Failed to load transactions.json: %s", e)
    TRANSACTIONS = {}

try:
    with open(DATA_DIR / "incidents.json", encoding="utf-8") as f:
        INCIDENTS: list[dict] = json.load(f)
except Exception as e:
    logger.error("Failed to load incidents.json: %s", e)
    INCIDENTS = []


def transaction_lookup(tx_id: str) -> dict:
    """Look up a transaction by ID."""
    try:
        key = tx_id.strip().upper()
        if key in TRANSACTIONS:
            return dict(TRANSACTIONS[key])
        return {"error": f"Transaction {tx_id} not found"}
    except Exception as e:
        logger.error("transaction_lookup failed: %s", e)
        return {"error": f"Lookup failed: {e}"}


def incident_search(query: str) -> dict:
    """Search incidents by keyword matching."""
    try:
        keywords = [k.lower() for k in re.split(r'\W+', query) if len(k) > 2]
        if not keywords:
            return {"results": [], "count": 0, "message": "No matching incidents found"}

        scored = []
        for inc in INCIDENTS:
            text = " ".join([
                inc.get("title", ""),
                inc.get("description", ""),
                inc.get("error_code", ""),
                inc.get("resolution", ""),
            ]).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, inc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:3]]

        if not results:
            return {"results": [], "count": 0, "message": "No matching incidents found"}
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("incident_search failed: %s", e)
        return {"error": f"Search failed: {e}"}


def calculator(expression: str) -> dict:
    """Calculate arithmetic or percentage expressions safely (no eval)."""
    try:
        expr = expression.strip()

        # Pattern: X% of Y
        m = re.match(r'([\d.]+)\s*%\s*of\s*[₹]?\s*([\d,]+(?:\.\d+)?)', expr, re.IGNORECASE)
        if m:
            pct = float(m.group(1))
            val = float(m.group(2).replace(",", ""))
            result = val * pct / 100
            return {"result": result, "expression": expression, "formatted": f"\u20b9{result:,.2f}"}

        # Pattern: X op Y
        m = re.match(r'[₹]?\s*([\d,]+(?:\.\d+)?)\s*([+\-*/])\s*[₹]?\s*([\d,]+(?:\.\d+)?)', expr)
        if m:
            a = float(m.group(1).replace(",", ""))
            op = m.group(2)
            b = float(m.group(3).replace(",", ""))
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            elif op == "/":
                if b == 0:
                    return {"error": "Division by zero"}
                result = a / b
            else:
                return {"error": "Unsupported operator"}
            return {"result": result, "expression": expression, "formatted": f"\u20b9{result:,.2f}"}

        return {"error": "Cannot parse expression"}
    except Exception as e:
        logger.error("calculator failed: %s", e)
        return {"error": f"Calculation failed: {e}"}


def refund_transaction(tx_id: str, reason: str = "") -> dict:
    """Simulate a refund (modifies in-memory state only)."""
    try:
        key = tx_id.strip().upper()
        if key not in TRANSACTIONS:
            return {"error": f"Transaction {tx_id} not found"}
        tx = TRANSACTIONS[key]
        if tx.get("status") == "refunded":
            return {"error": f"Transaction {tx_id} is already refunded"}
        previous_status = tx["status"]
        tx["status"] = "refunded"
        tx["refunded_at"] = datetime.now(timezone.utc).isoformat()
        if reason:
            tx["refund_reason"] = reason
        return {
            "success": True,
            "tx_id": tx_id.upper(),
            "previous_status": previous_status,
            "message": "Simulated refund applied (mock only)",
        }
    except Exception as e:
        logger.error("refund_transaction failed: %s", e)
        return {"error": f"Refund failed: {e}"}


def run_tool(tool_name: str, query: str) -> dict:
    """Dispatch to the correct tool based on tool_name."""
    try:
        tx_match = re.search(r'TX-\d+', query, re.IGNORECASE)
        tx_id = tx_match.group(0).upper() if tx_match else ""

        if tool_name == "transaction_lookup":
            if not tx_id:
                return {"error": "No transaction ID found in query"}
            return transaction_lookup(tx_id)

        elif tool_name == "incident_search":
            return incident_search(query)

        elif tool_name == "calculator":
            return calculator(query)

        elif tool_name == "refund_transaction":
            if not tx_id:
                return {"error": "No transaction ID found in query"}
            return refund_transaction(tx_id)

        elif tool_name in ("none", "low_confidence"):
            return {}

        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        logger.error("run_tool failed: %s", e)
        return {"error": f"Tool dispatch failed: {e}"}
