"""Offline tests: no API key, no network. Run with `pytest` or `python tests/test_retrieval.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bot import HelpdeskBot, clean_history  # noqa: E402
from app.config import settings  # noqa: E402
from app.prompts import build_user_message  # noqa: E402
from app.retrieval import BM25Index, load_chunks, stem, tokenize  # noqa: E402

CHUNKS = load_chunks(settings.kb_dir)
INDEX = BM25Index(CHUNKS)


def top(query: str):
    hits = INDEX.search(query, k=3, min_score=settings.min_score, min_matched=settings.min_matched)
    return hits[0].chunk.source if hits else None


def test_chunker_splits_by_section():
    assert len(CHUNKS) > 30
    vpn = [c for c in CHUNKS if c.source == "vpn.md"]
    assert [c.heading for c in vpn] == ["When you need the VPN", "Common VPN errors", "Session limits"]
    assert all(c.text for c in CHUNKS)


def test_stemming_and_synonyms():
    assert stem("resetting") == "reset"
    assert stem("passwords") == "password"
    assert stem("policies") == "policy"
    assert tokenize("Wi-Fi") == tokenize("wifi") == ["wifi"]
    assert "laptop" in tokenize("my computer is slow")


def test_finds_the_right_document():
    assert top("how do I reset my password") == "password-and-mfa.md"
    assert top("vpn error 809") == "vpn.md"
    assert top("how do I release a print job") == "printers-and-scanning.md"
    assert top("I clicked a phishing link") == "security-and-phishing.md"


def test_abstains_when_nothing_relevant():
    assert top("who won the cricket match") is None
    assert top("cafeteria menu today") is None


def test_history_is_cleaned():
    messy = [
        {"role": "assistant", "content": "stray"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert clean_history(messy, 6) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_short_followups_use_previous_question():
    bot = HelpdeskBot(client=object())
    history = [{"role": "user", "content": "my vpn error 809"}, {"role": "assistant", "content": "Switch to TCP."}]
    hits = bot.retrieve("and on mac?", history)
    assert hits and hits[0].chunk.source == "vpn.md"


def test_abstention_makes_no_model_call():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("model must not be called when retrieval finds nothing")

    result = HelpdeskBot(client=Boom()).answer("who won the cricket match")
    assert result["abstained"] and not result["sources"] and "not sure" in result["answer"].lower()


def test_prompt_wraps_documents_as_data():
    hits = INDEX.search("vpn error 809", k=1)
    message = build_user_message("vpn error 809", hits)
    assert message.startswith("<documents>") and '<document index="1"' in message and "<question>" in message


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
