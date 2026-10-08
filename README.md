# PaymentOps Agent — Decision Model + LLM for Payment Operations

A working demo that shows how an open-source decision model (Laya) can handle bounded classifications around an LLM instead of letting the LLM decide everything. Built with Python, LangGraph, and Streamlit on local mock payment data — no real payment APIs involved.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Laya Tool Router      │  ← structured choice: which tool?
│   (decision model)      │
└──────────┬──────────────┘
           │
    ┌──────┴──────────────┐
    │                     │
    ▼                     ▼
transaction_lookup    refund_transaction
incident_search           │
calculator                ▼
    │          ┌──────────────────────┐
    │          │  Laya Action Gate    │  ← safe / requires_approval / blocked
    │          │  (decision model)    │
    │          └──────────┬───────────┘
    │                     │
    │          ┌──────────┴──────────┐
    │          │                     │
    │      (safe only)        (blocked/approval)
    │          │                     │
    └──────────┘                     │
         │                           │
         ▼                           ▼
   Tool Execution             Skip execution
         │                           │
         └──────────┬────────────────┘
                    ▼
         ┌──────────────────┐
         │   LLM Response   │  ← summarises tool result in natural language
         └──────────────────┘
```

**Data flow rule:** Laya makes every routing and gating decision. The LLM only generates the final user-facing text.

---

## Why Not LLM for Everything?

| Concern | LLM approach | Laya + LLM approach |
|---|---|---|
| Latency | Every decision pays LLM inference cost | Routing is a fast local model call |
| Determinism | LLM tool-calling can hallucinate tool names | Laya returns a typed `choice` from a fixed label set |
| Auditability | Hard to log *why* a tool was chosen | Laya returns per-label probabilities — trace is exact |
| Cost | Repeated LLM calls per routing step | LLM called once, only for generation |
| Safety | LLM may approve a write action it shouldn't | Gating is a separate, inspectable Laya call |

The LLM is good at one thing here: turning a structured tool result into a readable, empathetic answer. It is bad at the thing Laya does: making fast, auditable, bounded decisions from a fixed label set.

---

## Where Laya Is Used

### Tool Routing

When the user submits a query, Laya receives the query text plus a `choice` question with five labelled criteria:

```
transaction_lookup  — retrieve details about a specific transaction
incident_search     — search runbooks and known error patterns
calculator          — compute refund/fee/settlement values
refund_transaction  — initiate a refund
none                — general queries with no matching tool
```

Laya returns the winning label and a per-label probability distribution. If the winning label's confidence is below the configured threshold (default 0.60), the agent falls back to asking the user for clarification rather than guessing.

### Action Gating

Whenever Laya routes to `refund_transaction`, a second Laya call evaluates whether the action should proceed:

```
safe               — straightforward, low-risk refund
requires_approval  — needs human review before execution
blocked            — suspicious or explicitly disallowed
```

Only a `safe` decision lets the refund tool actually run. Everything else routes to the LLM to explain the outcome without executing any write.

---

## Example Decision Traces

### Refund requiring approval

```
USER
Can you refund TX-1042?

LAYA — TOOL ROUTER
Decision: refund_transaction
Confidence: 0.91

LAYA — ACTION GATE
Decision: requires_approval
Confidence: 0.88

SYSTEM
Refund NOT executed. Gate decision: requires_approval.

LLM
TX-1042 is eligible for review, but the refund was not executed
because this action requires approval. Please await confirmation
from a supervisor before this transaction can be processed.
```

### Transaction lookup

```
USER
What amount was charged for TX-1042?

LAYA — TOOL ROUTER
Decision: transaction_lookup
Confidence: 0.94

TOOL EXECUTION
{ "id": "TX-1042", "amount": 4999.0, "currency": "INR",
  "status": "failed", "error_code": "E003",
  "merchant": "Paytm Mall", ... }

LLM
Transaction TX-1042 was a failed payment of ₹4,999.00 to Paytm Mall.
It was declined by the issuer (error code E003) on 1 October 2024.
```

### Low-confidence fallback

```
USER
Do the thing with that payment

LAYA — TOOL ROUTER
Decision: transaction_lookup
Confidence: 0.43   ← below 0.60 threshold

SYSTEM
Routing confidence 43% is below threshold 60%. Falling back to clarification.

LLM
I wasn't able to confidently understand your request. Could you
provide a transaction ID or tell me whether you'd like to look up
a transaction, search for an error, or calculate something?
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd paymentops-agent

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key for the LLM response step |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name to use for response generation |
| `ROUTING_CONFIDENCE_THRESHOLD` | `0.60` | Minimum Laya confidence to act on a routing decision |
| `LAYA_DEVICE` | `cpu` | Device for Laya inference (`cpu` or `cuda`) |

---

## Running

Start the Streamlit app:

```bash
streamlit run app.py
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Known Limitations

- **Mock data only.** All transactions and refunds are in-memory. Restarting the app resets any simulated refunds.
- **Laya model size.** The first run downloads the Laya checkpoint. A cold start may take a few seconds on CPU.
- **LLM key required for responses.** Without a valid `OPENAI_API_KEY`, the agent still routes and gates correctly but returns a warning instead of a natural-language answer.
- **Single-turn only.** The UI maintains a conversation history for display, but each query is processed independently — there is no multi-turn memory passed to the LLM.
- **No authentication.** This is a local demo. Do not expose it to the public internet without adding auth.
- **Calculator scope.** The expression parser handles `X% of Y` and basic `a ± * / b` forms. Complex nested arithmetic is not supported.

---

## Technical Notes

The project deliberately separates three layers:

**Laya → fast bounded decisions.** Laya is a small, local decision model that takes a structured question (a `choice` with labelled criteria) and returns a typed answer with a confidence score. It never generates free text. This makes every routing and gating decision fast, auditable, and deterministic given the same input.

**LLM → reasoning and generation.** The LLM (GPT-4o-mini by default) receives structured context — the user query, the tool result, and any gate outcome — and generates a concise, human-readable response. It is called exactly once per request, only after all decisions have been made.

**Python/tools → deterministic execution.** The four tools (`transaction_lookup`, `incident_search`, `calculator`, `refund_transaction`) are plain Python functions operating on local JSON data. They have no probabilistic behaviour and return structured dicts that both Laya (for gating context) and the LLM (for summarisation) can consume.

This separation means you can swap the LLM provider without touching the routing logic, tune the confidence threshold without retraining anything, and audit every decision via the probability trace visible in the UI.
