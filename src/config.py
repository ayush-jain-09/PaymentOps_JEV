import os
from dotenv import load_dotenv

load_dotenv()

ROUTING_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.60"))
LAYA_DEVICE: str = os.getenv("LAYA_DEVICE", "cpu")

# ── LLM provider configuration ────────────────────────────────────────────────
# Supported values: "openrouter"
# Add more providers in src/llm.py without changing anything else.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")

# OpenRouter
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
