# IT Helpdesk Assistant (RAG chatbot with citations and an evaluation harness)

A support chatbot that answers employees' IT questions **only from a company knowledge base**, cites the
documents it used, and **declines** when the knowledge base has no answer. It includes a streaming web UI,
an offline retrieval test suite and an evaluation script with a labelled question set.

The company ("Northwind Labs") and its documents are fictional. To use your own content, replace the
markdown files in `data/kb/`.

> Add a screenshot or short GIF of the chat here, and a link to your live demo.

## Why this project

Support bots fail in two ways: they invent answers, and nobody measures how often. This project is built
around those two problems:

- **Grounding:** every answer is generated from retrieved passages and cites them as `[1]`, `[2]`.
- **Abstention:** if retrieval finds nothing relevant, the bot says so *without calling the model at all*.
- **Measurement:** a labelled question set and a script report retrieval accuracy, fact accuracy, citation
  rate and abstention rate, so changes can be compared with numbers instead of impressions.

## Architecture

```
Browser (static/index.html)
   |  POST /api/chat  (message + recent history)
   v
FastAPI (app/main.py)  -- validation, rate limiting, server-sent events
   |
   v
HelpdeskBot (app/bot.py)
   |-- 1. retrieve: BM25 over markdown sections (app/retrieval.py)
   |-- 2. abstain: no relevant chunk -> canned "not sure" reply, no model call
   |-- 3. generate: Claude answers from <documents>, cites [n]   (app/prompts.py)
   v
Streamed events: sources -> tokens -> done
```

| Part | File | Notes |
|---|---|---|
| Chunking + search | `app/retrieval.py` | One chunk per `##` section; BM25 with title and heading boosts; light stemmer and synonym map |
| Prompts | `app/prompts.py` | System rules, document wrapping, canned abstention message |
| Orchestration | `app/bot.py` | Retrieval, abstention, history cleaning, streaming, error handling |
| API | `app/main.py` | `POST /api/chat` (SSE), `GET /api/health`, rate limit, input limits |
| UI | `static/index.html` | Streams tokens, renders `[n]` citations and a sources list without `innerHTML` |
| Tests | `tests/test_retrieval.py` | 8 offline tests, no API key needed |
| Evaluation | `eval/run_eval.py`, `eval/questions.jsonl` | 36 answerable and 6 out-of-scope questions |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

Docker: `docker build -t helpdesk-bot . && docker run -p 8000:8000 --env-file .env helpdesk-bot`

Settings (environment variables): `ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`), `TOP_K`,
`MIN_SCORE`, `MIN_MATCHED`, `COMPANY_NAME`, `SUPPORT_EMAIL`. See `app/config.py`.

## Evaluation

```bash
python tests/test_retrieval.py            # or: pytest
python eval/run_eval.py --retrieval-only  # no API key needed
python eval/run_eval.py                   # full pipeline, needs ANTHROPIC_API_KEY
```

### Retrieval results (measured, no model involved)

| Metric | Result |
|---|---|
| Retrieval hit@4 (expected document in top 4) | 29 / 36 (81%) |
| Out-of-scope questions filtered by retrieval | 5 / 6 (83%) |

### End-to-end results (run this yourself and fill in)

| Metric | Result |
|---|---|
| Fact match | _run `python eval/run_eval.py`_ |
| Cited a source | |
| Cited the expected document | |
| Correct abstention | |

### What the retrieval failures show

The 7 misses are all vocabulary mismatches, which is the known weakness of keyword search:

- "How often do I have to **change** my password?" vs. a document that says passwords **expire**.
- "**Wrong password too many times**" vs. "**failed sign-in attempts**".
- "Can I plug in a **USB stick**?" vs. "**USB storage** is blocked".
- "Opening **hours**" vs. "the helpdesk is **staffed** Monday to Friday".

The next step would be hybrid retrieval (BM25 plus embeddings) with a reranker, then re-running this same
evaluation to see whether hit@4 improves. The question set is small and was written by the author, so treat
the numbers as a baseline for comparing changes, not as a benchmark.

## Design decisions

- **BM25 instead of embeddings.** Anthropic does not offer an embeddings endpoint, and for a knowledge base of
  about 45 sections a keyword index is fast, free, dependency-free and easy to debug. The retriever sits behind
  one function (`BM25Index.search`), so swapping in embeddings or a hybrid is a contained change.
- **Abstain before generating.** Requiring at least two matching query terms and a minimum score stops the
  bot from answering unrelated questions and saves a model call. The trade-off is that some valid questions
  with unusual wording are declined, which the evaluation makes visible.
- **Small, cheap model by default.** Answers are short and grounded, so a fast model is enough. Change it with
  `ANTHROPIC_MODEL`.
- **Prompt-injection hygiene.** Retrieved text and the user's message are wrapped as data, the system prompt
  says to ignore instructions inside them, and the bot has no tools, so it cannot take actions even if
  manipulated. The UI renders model output with `textContent`, never as HTML.
- **No actions.** The bot explains steps; it cannot reset passwords or look up tickets. Adding tool use with
  authentication would be a separate, larger project.

## Limitations

- Keyword retrieval misses paraphrases (see above); the synonym map is hand-written.
- Follow-up handling is simple: short questions are searched together with the previous question.
- The rate limiter is in-memory and per process; use a shared store or a gateway in production.
- There is no user authentication, conversation storage or analytics.
- The knowledge base is synthetic, so real-world messiness (long documents, tables, outdated pages) is not
  represented.

## Roadmap

1. Hybrid retrieval (BM25 plus embeddings) and a reranker; compare against the current baseline.
2. An LLM-as-judge check for answer faithfulness, added to `eval/run_eval.py`.
3. Thumbs up/down feedback in the UI, logged to review failures.
4. Ticket creation handoff when the bot cannot answer.
