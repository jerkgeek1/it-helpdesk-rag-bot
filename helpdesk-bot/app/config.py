"""Settings, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader so the project has no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()


@dataclass(frozen=True)
class Settings:
    kb_dir: Path = ROOT / "data" / "kb"
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "400"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "4.0"))
    min_matched: int = int(os.getenv("MIN_MATCHED", "2"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
    rate_limit_window_s: int = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
    support_email: str = os.getenv("SUPPORT_EMAIL", "helpdesk@northwind.example")
    company: str = os.getenv("COMPANY_NAME", "Northwind Labs")


settings = Settings()
