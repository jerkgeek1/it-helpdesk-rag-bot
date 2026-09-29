"""Chunking and BM25 retrieval over a folder of markdown knowledge-base files.

Pure Python on purpose: no vector database or embedding service is needed for a
knowledge base of this size, and every retrieval decision can be inspected and
unit-tested offline. See the README for when to move to embeddings.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for with from as is are was were be been being
    do does did done doing have has had having i me my we our you your it its this that these those
    can could should would will shall may might must what which who whom whose when where why how
    there here about into out up down over under again not no yes any some all more most other
    please get got need want just also than too very""".split()
)

# Small domain synonym map, applied before stemming.
SYNONYMS = {
    "2fa": "mfa",
    "authenticator": "mfa",
    "passcode": "password",
    "pwd": "password",
    "notebook": "laptop",
    "computer": "laptop",
    "pc": "laptop",
    "macbook": "laptop",
    "thinkpad": "laptop",
    "internet": "network",
    "wireless": "wifi",
    "mail": "email",
    "inbox": "email",
    "outlook": "email",
    "installing": "install",
    "download": "install",
    "app": "software",
    "application": "software",
    "program": "software",
    "vpn": "vpn",
    "hotspot": "wifi",
    "recording": "recording",
    "record": "recording",
    "scam": "phishing",
    "spam": "phishing",
    "hack": "phishing",
    "hacked": "phishing",
    "stolen": "lost",
    "missing": "lost",
    "slowly": "slow",
    "sluggish": "slow",
    "freezing": "slow",
    "freezes": "slow",
    "camera": "camera",
    "webcam": "camera",
    "mic": "microphone",
    "cancel": "cancel",
    "refresh": "refresh",
}


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"wi[\s-]?fi", "wifi", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"log[\s-]?in", "signin", text)
    text = re.sub(r"sign[\s-]?in", "signin", text)
    text = re.sub(r"multi[\s-]?factor", "mfa", text)
    text = re.sub(r"two[\s-]?factor", "mfa", text)
    return text


def stem(word: str) -> str:
    """A deliberately light stemmer: enough to match plurals and -ing/-ed forms."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]  # "resetting" -> "reset"
            return word
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", _normalise(text)):
        if raw in STOPWORDS:
            continue
        token = stem(SYNONYMS.get(raw, raw))
        if token not in STOPWORDS:  # stemming can turn a word into a stopword ("gets" -> "get")
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name, e.g. "vpn.md"
    title: str  # document title (the "# " heading)
    heading: str  # section title (the "## " heading)
    text: str  # section body

    @property
    def prompt_text(self) -> str:
        return f"{self.title} - {self.heading}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    matched_terms: int


def load_chunks(kb_dir: Path) -> list[Chunk]:
    """Split every *.md file into one chunk per '## ' section."""
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        title = path.stem
        heading = "Overview"
        body: list[str] = []
        section = 0

        def flush() -> None:
            nonlocal section
            text = "\n".join(body).strip()
            if text:
                section += 1
                chunks.append(Chunk(f"{path.stem}#{section}", path.name, title, heading, text))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                heading = line[3:].strip()
                body = []
            else:
                body.append(line)
        flush()
    return chunks


class BM25Index:
    """Okapi BM25 over chunks, with title and heading terms boosted."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: Counter = Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.title) * 2 + tokenize(chunk.heading) * 3 + tokenize(chunk.text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._len.append(len(tokens))
            df.update(counts.keys())
        n = len(chunks)
        self._avg_len = sum(self._len) / n if n else 0.0
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.0,
        min_matched: int = 1,
    ) -> list[Hit]:
        """Return up to k chunks. Chunks below min_score, or matching fewer than
        min_matched distinct query terms, are dropped so the bot can abstain."""
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        needed = min(min_matched, len(terms))
        hits: list[Hit] = []
        for i, chunk in enumerate(self.chunks):
            tf = self._tf[i]
            score, matched = 0.0, 0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                matched += 1
                norm = freq + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
                score += self._idf[term] * freq * (self.k1 + 1) / norm
            if matched >= needed and score > min_score:
                hits.append(Hit(chunk, score, matched))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
