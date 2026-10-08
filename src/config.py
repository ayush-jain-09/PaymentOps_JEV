import os
from dotenv import load_dotenv

load_dotenv()

ROUTING_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.60"))
LAYA_DEVICE: str = os.getenv("LAYA_DEVICE", "cpu")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
