"""Evaluate the helpdesk bot on eval/questions.jsonl.

Two modes:
  python eval/run_eval.py --retrieval-only   no API key needed; measures retrieval
  python eval/run_eval.py                    full pipeline; needs ANTHROPIC_API_KEY

Metrics
  Retrieval hit@k      answerable questions whose expected document appears in the top-k
  Correct abstention   out-of-scope questions where the bot declines instead of answering
  Fact match           answerable questions whose answer contains every required fact
  Cited                answerable questions whose answer cites at least one source [n]
  Cited expected doc   answerable questions whose citations include the expected document

These are cheap, deterministic checks. They do not judge tone or completeness, so read the
failures list, and consider adding an LLM judge for a deeper evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.bot import HelpdeskBot  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results" / "latest.json"
ABSTAIN_MARKERS = ("not sure", "can't help", "cannot help", "only help with it")


def load_questions(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def run(retrieval_only: bool, limit: int | None) -> int:
    bot = HelpdeskBot()
    questions = load_questions(limit)
    answerable = [q for q in questions if q["answerable"]]
    out_of_scope = [q for q in questions if not q["answerable"]]
    rows: list[dict] = []
    failures: list[str] = []

    hit_count = 0
    for q in answerable:
        hits = bot.retrieve(q["question"])
        sources = [h.chunk.source for h in hits]
        hit = q["expected_source"] in sources
        hit_count += hit
        rows.append({"id": q["id"], "retrieved": sources, "retrieval_hit": hit})
        if not hit:
            failures.append(f"[retrieval] {q['id']}: expected {q['expected_source']}, got {sources or 'nothing'}")

    retrieval_abstain = 0
    for q in out_of_scope:
        hits = bot.retrieve(q["question"])
        retrieval_abstain += not hits
        rows.append({"id": q["id"], "retrieved": [h.chunk.source for h in hits], "retrieval_abstained": not hits})

    print(f"Questions: {len(answerable)} answerable, {len(out_of_scope)} out of scope")
    print(f"Retrieval hit@{bot.cfg.top_k}:            {pct(hit_count, len(answerable))}")
    print(f"Out-of-scope filtered by retrieval: {pct(retrieval_abstain, len(out_of_scope))}")

    summary = {
        "k": bot.cfg.top_k,
        "model": None if retrieval_only else bot.cfg.model,
        "retrieval_hit": [hit_count, len(answerable)],
        "retrieval_abstain": [retrieval_abstain, len(out_of_scope)],
    }

    if not retrieval_only:
        facts_ok = cited = cited_expected = correct_abstain = 0
        by_id = {r["id"]: r for r in rows}
        for i, q in enumerate(questions, start=1):
            result = bot.answer(q["question"])
            text = result["answer"]
            row = by_id[q["id"]]
            row["answer"] = text
            if result["error"]:
                failures.append(f"[api] {q['id']}: {result['error']}")
                continue
            if q["answerable"]:
                lowered = text.lower()
                ok = all(fact.lower() in lowered for fact in q["must_include"])
                nums = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
                cited_docs = {s["source"] for s in result["sources"] if s["n"] in nums}
                facts_ok += ok
                cited += bool(nums)
                cited_expected += q["expected_source"] in cited_docs
                row.update(fact_match=ok, cited=bool(nums), cited_expected=q["expected_source"] in cited_docs)
                if not ok:
                    failures.append(f"[facts] {q['id']}: missing {q['must_include']} in: {text[:160]!r}")
            else:
                abstained = result["abstained"] or any(m in text.lower() for m in ABSTAIN_MARKERS)
                correct_abstain += abstained
                row["abstained"] = abstained
                if not abstained:
                    failures.append(f"[abstain] {q['id']}: answered instead of declining: {text[:160]!r}")
            print(f"  {i}/{len(questions)} {q['id']}", end="\r", flush=True)
            time.sleep(0.2)
        print(" " * 40)
        print(f"Fact match:                         {pct(facts_ok, len(answerable))}")
        print(f"Cited a source:                     {pct(cited, len(answerable))}")
        print(f"Cited the expected document:        {pct(cited_expected, len(answerable))}")
        print(f"Correct abstention (end to end):    {pct(correct_abstain, len(out_of_scope))}")
        summary.update(
            fact_match=[facts_ok, len(answerable)],
            cited=[cited, len(answerable)],
            cited_expected=[cited_expected, len(answerable)],
            correct_abstain=[correct_abstain, len(out_of_scope)],
        )

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print("  " + f)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "rows": rows, "failures": failures}, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the model; no API key needed")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()
    sys.exit(run(args.retrieval_only, args.limit))
