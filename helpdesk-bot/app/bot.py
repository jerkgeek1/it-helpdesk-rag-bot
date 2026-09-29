"""The helpdesk bot: retrieve, decide whether to answer, then ask Claude.

Two entry points share the same logic:
  * HelpdeskBot.stream()  - yields events for the web UI (server-sent events)
  * HelpdeskBot.answer()  - returns one dict, used by the evaluation script
"""
from __future__ import annotations

import logging
from typing import Iterator

from .config import Settings, settings as default_settings
from .prompts import NO_ANSWER, build_system_prompt, build_user_message
from .retrieval import BM25Index, Hit, load_chunks, tokenize

log = logging.getLogger("helpdesk")


def clean_history(history: list[dict], max_turns: int) -> list[dict]:
    """Keep only strictly alternating user/assistant turns, starting with a user turn
    and ending with an assistant turn, limited to the most recent max_turns messages."""
    cleaned: list[dict] = []
    expected = "user"
    for turn in history:
        if turn.get("role") == expected and str(turn.get("content", "")).strip():
            cleaned.append({"role": expected, "content": str(turn["content"])})
            expected = "assistant" if expected == "user" else "user"
    if cleaned and cleaned[-1]["role"] == "user":
        cleaned.pop()
    cleaned = cleaned[-max_turns:]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


class HelpdeskBot:
    def __init__(self, cfg: Settings = default_settings, client=None) -> None:
        self.cfg = cfg
        self.index = BM25Index(load_chunks(cfg.kb_dir))
        self._client = client
        self._system = build_system_prompt(cfg.company, cfg.support_email)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, message: str, history: list[dict] | None = None) -> list[Hit]:
        """Search with the message; for very short follow-ups ("and on Mac?") also
        include the previous user question so the search has enough context."""
        query = message
        if len(tokenize(message)) < 3 and history:
            previous = [t["content"] for t in history if t["role"] == "user"]
            if previous:
                query = previous[-1] + " " + message
        return self.index.search(
            query, k=self.cfg.top_k, min_score=self.cfg.min_score, min_matched=self.cfg.min_matched
        )

    # -- Claude ------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so retrieval and tests run without the SDK

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def _messages(self, message: str, hits: list[Hit], history: list[dict]) -> list[dict]:
        return [*history, {"role": "user", "content": build_user_message(message, hits)}]

    @staticmethod
    def _sources(hits: list[Hit]) -> list[dict]:
        return [
            {"n": i, "source": h.chunk.source, "title": h.chunk.title, "section": h.chunk.heading}
            for i, h in enumerate(hits, start=1)
        ]

    # -- public API --------------------------------------------------------
    def stream(self, message: str, history: list[dict] | None = None) -> Iterator[dict]:
        history = clean_history(history or [], self.cfg.max_history_turns)
        hits = self.retrieve(message, history)
        log.info("query=%r hits=%s", message, [(h.chunk.id, round(h.score, 1)) for h in hits])

        if not hits:  # abstain without spending a model call
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": NO_ANSWER.format(email=self.cfg.support_email)}
            yield {"type": "done", "abstained": True}
            return

        yield {"type": "sources", "sources": self._sources(hits)}
        try:
            with self._get_client().messages.stream(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=0.2,
                system=self._system,
                messages=self._messages(message, hits, history),
            ) as stream:
                for text in stream.text_stream:
                    yield {"type": "token", "text": text}
        except Exception as exc:  # network, auth, rate limit, overloaded...
            log.exception("model call failed")
            yield {
                "type": "error",
                "message": f"The assistant is unavailable right now ({type(exc).__name__}). "
                f"Please email {self.cfg.support_email}.",
            }
            return
        yield {"type": "done", "abstained": False}

    def answer(self, message: str, history: list[dict] | None = None) -> dict:
        """Non-streaming version for evaluation."""
        text, sources, abstained, error = "", [], False, None
        for event in self.stream(message, history):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                text += event["text"]
            elif event["type"] == "done":
                abstained = event["abstained"]
            elif event["type"] == "error":
                error = event["message"]
        return {"answer": text, "sources": sources, "abstained": abstained, "error": error}
