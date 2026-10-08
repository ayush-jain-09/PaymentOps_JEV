import pytest
from src.tools import transaction_lookup, incident_search, calculator, refund_transaction, TRANSACTIONS


def test_transaction_lookup_found():
    result = transaction_lookup("TX-1001")
    assert "id" in result
    assert result["id"] == "TX-1001"


def test_transaction_lookup_tx1042():
    result = transaction_lookup("TX-1042")
    assert "id" in result
    assert result["id"] == "TX-1042"


def test_transaction_lookup_not_found():
    result = transaction_lookup("TX-9999")
    assert "error" in result


def test_incident_search_timeout():
    result = incident_search("timeout")
    assert result.get("count", 0) >= 1


def test_incident_search_no_match():
    result = incident_search("xyzzy")
    assert result.get("count", 0) == 0


def test_calculator_percentage():
    result = calculator("80% of 4999")
    assert "result" in result
    assert abs(result["result"] - 3999.2) < 0.01


def test_calculator_subtraction():
    result = calculator("5000 - 4750")
    assert "result" in result
    assert abs(result["result"] - 250.0) < 0.01


def test_calculator_invalid():
    result = calculator("hello world")
    assert "error" in result


def test_refund_simulated():
    # Reset TX-1005 in memory in case a previous test already refunded it
    if "TX-1005" in TRANSACTIONS:
        TRANSACTIONS["TX-1005"]["status"] = "completed"
        TRANSACTIONS["TX-1005"].pop("refunded_at", None)
    result = refund_transaction("TX-1005")
    assert result.get("success") is True


def test_refund_not_found():
    result = refund_transaction("TX-9999")
    assert "error" in result
