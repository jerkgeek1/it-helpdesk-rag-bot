"""FastAPI app: a streaming chat endpoint plus the static web UI."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bot import HelpdeskBot
from .config import ROOT, settings

app = FastAPI(title=f"{settings.company} IT helpdesk bot")
bot = HelpdeskBot()

# --- tiny in-memory rate limiter (per client IP). Use Redis or an API gateway in production.
_requests: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client_id: str) -> None:
    now = time.monotonic()
    window = _requests[client_id]
    while window and now - window[0] > settings.rate_limit_window_s:
        window.popleft()
    if len(window) >= settings.rate_limit_requests:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    window.append(now)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=12)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(bot.index.chunks),
        "model": settings.model,
        "api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    check_rate_limit(request.client.host if request.client else "unknown")
    history = [t.model_dump() for t in req.history]

    def events():
        for event in bot.stream(req.message.strip(), history):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
