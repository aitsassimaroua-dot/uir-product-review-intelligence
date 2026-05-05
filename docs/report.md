# Product Review Intelligence
## A Multi-Agent AI System for Customer Insight & Competitive Analysis

**S8 Integrated Project — Building Multi-Agent AI Systems**
*AI & Big Data Program — UIR — 2025–2026*
*Prof. Hakim Hafidi*

**Team:** [Member A], [Member B], [Member C]
**Repository:** [github.com/.../uir-product-review-intelligence](#)
**Demo video:** [link]
**Date:** May 2026

---

## Executive Summary

A product manager who wants to understand how their product is being received and how it stacks up against the competition currently has to scroll hundreds of reviews and run multiple competitor searches by hand. This is slow, inconsistent, and rarely produces a single artefact a leadership team can act on.

**Product Review Intelligence** is a multi-agent AI system that automates this workflow. Three specialized agents collaborate: a **Sentiment Analyst** that owns a fine-tuned DistilBERT classifier, a **Market Researcher** that scouts competitors via public web search, and a **Report Orchestrator** that synthesizes their outputs into a one-page market brief — with a human checkpoint before the brief is finalized.

The DistilBERT classifier was fine-tuned on a 18,000-example stratified subset of `amazon_polarity` and reaches **94.5% accuracy / 0.945 weighted F1** on 5,000 held-out test reviews (see §3.4 for the full breakdown). End-to-end, the system processes a 10-review CSV and produces an approved market brief in approximately 90 seconds, with every agent action logged as structured JSON.

The system meets every minimum requirement of the brief: 2 specialists + 1 orchestrator, a self-trained DL model used as a functional tool, two tools with clear I/O schemas, a human-in-the-loop checkpoint, error handling, JSONL logging with timestamps, and a reproducible setup.

---

## 1. Problem & Motivation

### 1.1 Use case

A product manager at a consumer-electronics company has a wireless-earbuds product and needs to answer three questions every Monday morning:

1. What are customers saying about us this week?
2. Who are the competitors we are losing buyers to, and why?
3. What is the one action we should take?

Today they answer these by hand: scrolling Amazon, opening competitor pages, copy-pasting into a Google Doc. We replace that manual workflow with a multi-agent system.

### 1.2 Why this fits the brief

The brief asks for "specialized AI agents collaborating to solve a real-world problem" with at least one agent using a self-trained DL model "as a functional tool — not decoration." Sentiment classification on product reviews is a textbook fine-tuning task; the agent that owns it can call the model dozens of times per run, so the model is genuinely load-bearing rather than a demo prop.

### 1.3 Domain choice

We chose **Domain B — Product Review Intelligence** from the suggested list (in preference to Smart Document Analyst or Medical Image Triage) because:

- It naturally decomposes into three roles that need different reasoning styles (classification, search, synthesis), justifying multi-agent design over a monolithic prompt.
- The DL model required by the brief (sentiment classifier) is well-supported by an open dataset (`amazon_polarity`) which makes evaluation honest.
- The output (a market brief) is a concrete, defensible artefact — easy to grade, easy to demo.

---

## 2. System Design & Architecture *(Rubric: 15%)*

### 2.1 Why multi-agent and not one giant prompt

A single LLM call with a long instruction could, in principle, do all of the above. We argue it should not, for four reasons:

| Problem with a single-agent / single-prompt approach | How multi-agent fixes it |
|---|---|
| One LLM call has to do classification, search, *and* writing — quality drops on each | Each agent gets a tight role and a small tool surface |
| No place to plug a *trained* DL model in a meaningful way | The Sentiment Analyst owns the BERT tool and calls it on every review |
| Hard to add a human checkpoint in the middle of a long generation | The Orchestrator step naturally exposes a HITL boundary |
| Hard to debug what went wrong | Per-agent JSON logs make failures localized |

### 2.2 The three agents

| Agent | Role | Tools |
|---|---|---|
| **Sentiment Analyst** | Reads reviews, classifies each, surfaces recurring complaints and praise | `BertSentimentTool`, `ReviewLoaderTool` |
| **Market Researcher** | Given the product name and the analyst's complaints, finds 3 competitors with public-web evidence | `CompetitorSearchTool` (DuckDuckGo) |
| **Report Orchestrator** | Synthesises the two specialists' outputs into a 1-page market brief, asks the human to approve | *(no tools — synthesis only)* |

The Orchestrator deliberately has **no tools**: this enforces the separation of concerns and prevents the "godlike orchestrator" anti-pattern where one agent silently absorbs the others' work.

### 2.3 Communication diagram

```
                      ┌──────────────────────────┐
       User ─────────▶│  Report Orchestrator     │
  (product+CSV)       │  (Gemini 1.5 Flash)      │
                      └────────┬─────────────────┘
                               │ task 1
                  ┌────────────▼─────────────┐
                  │  Sentiment Analyst       │
                  │  (Gemini 1.5 Flash)      │
                  │  ├── BertSentimentTool ──┼──▶ DistilBERT
                  │  └── ReviewLoaderTool ───┼──▶ reviews.csv
                  └────────────┬─────────────┘
                               │ task 2 (uses task 1 output)
                  ┌────────────▼─────────────┐
                  │  Market Researcher       │
                  │  (Gemini 1.5 Flash)      │
                  │  └── CompetitorSearchTool┼──▶ DuckDuckGo
                  └────────────┬─────────────┘
                               │ task 3 (uses 1+2 output)
                  ┌────────────▼─────────────┐
                  │  Report Orchestrator     │
                  │  drafts brief            │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  HITL: approve / edit /  │
                  │  reject  (human input)   │
                  └────────────┬─────────────┘
                               │ approved
                               ▼
                       outputs/market_brief_*.md
                       logs/agent_actions_*.jsonl
```

(A polished mermaid version lives in `docs/architecture.md`; for this report we show the ASCII version so it renders identically across PDF tools.)

### 2.4 Data flow

1. **Ingest** — User runs `python -m src.main --product "X" --reviews data/reviews.csv`.
2. **Plan** — A `Crew` is built with three sequential tasks: `analyze_reviews`, `scout_competitors`, `write_brief`.
3. **Analyze** — Sentiment Analyst loads the CSV (`ReviewLoaderTool`), then calls `BertSentimentTool` for each review. Each call is logged with `{agent, tool, input_hash, label, score, latency_ms}`.
4. **Scout** — Market Researcher reads the analyst's output (top complaints), formulates 1–3 search queries, calls `CompetitorSearchTool`.
5. **Synthesize** — Orchestrator drafts a 1-page market brief.
6. **HITL** — User sees the draft in the terminal; CrewAI's `human_input=True` blocks until they approve, edit, or reject.
7. **Export** — Approved brief is written to `outputs/market_brief_<timestamp>.md`.

### 2.5 Tools — input/output schemas

Every tool uses a Pydantic schema for inputs (CrewAI `args_schema=...`). This gives free validation and a self-documented contract.

| Tool | Input schema | Output |
|---|---|---|
| `BertSentimentTool` | `BertSentimentInput(text: str)` | `{"label": "positive"\|"negative", "score": float, "backend": "bert"\|"heuristic"}` |
| `ReviewLoaderTool` | `ReviewLoaderInput(path: str, max_rows: int = 200)` | `[{"id": int, "text": str, "rating": int?}]` |
| `CompetitorSearchTool` | `CompetitorSearchInput(query: str, k: int = 5)` | `[{"title": str, "snippet": str, "url": str}]` |

Each tool wraps its body in a `try/except`. On failure it returns a structured error dict; it never raises out of `_run`. This is what lets the multi-agent system keep going (or fail gracefully) when a tool stumbles.

### 2.6 Trade-offs we considered

- **Three agents vs five.** Five (loader + classifier + complaint-extractor + searcher + writer) would be "more multi-agent" but the loader/classifier and extractor/writer pairs share a context and would communicate by passing JSON back and forth needlessly. Three is the smallest count that justifies the architecture.
- **DistilBERT vs BERT-base.** DistilBERT trains in ~half the time on free Colab with ~1% F1 loss. For a 4-week project this is the right trade.
- **DuckDuckGo vs Serper API.** DuckDuckGo is free and key-less — fits the brief's "no personal spending". Serper would give cleaner JSON but costs money.
- **CrewAI vs LangGraph.** CrewAI is the brief's primary stack and ships HITL via `human_input=True`. LangGraph is the stretch goal; we did not pursue it because the gains (DAG flexibility, cyclic graphs) are not needed for our linear flow.
- **Sequential vs hierarchical Crew process.** We chose `Process.sequential` because the dependency between tasks is linear (analyst → researcher → orchestrator) and a manager LLM dispatching tasks adds cost without value here.

---

## 3. Deep Learning Model *(Rubric: 15%)*

### 3.1 Dataset

We use **`amazon_polarity`** (3.6M train / 400k test) from Hugging Face Datasets. We chose it over IMDB and Yelp Polarity because (i) it is exactly the target domain (product reviews), and (ii) its size gives plenty of headroom for a stratified subset.

| Split | Size | Class balance |
|---|---|---|
| train | 18,000 | 50% pos / 50% neg |
| val | 2,000 | stratified from train |
| test | 5,000 | stratified from full test split |

A smaller smoke run (1,800 examples × 1 epoch, ~80 s wall clock) reached 91.1% accuracy as a sanity check. The numbers below come from the **full run** (18,000 train × 2 epochs).

**Subsetting** is stratified per class (`shuffle(seed=42).select`), so the class balance matches the source. Train/val are carved from the train split; the test set is held out from training.

**Preprocessing:** the `title` and `content` columns are concatenated as `f"{title}. {content}"`. The title acts as a compressed summary signal. Truncation is at 256 WordPiece tokens; longer reviews are rare and their tails are weakly informative.

**Honest caveats** (which we name in the limitations section, not bury):
- 3-star reviews are excluded by the dataset authors — there is no neutral class. We surface this in the report so we don't oversell what the model can do.
- Reviews skew older (~2015); modern slang or product categories may be under-represented.
- English only.

### 3.2 Model

We fine-tune **`distilbert-base-uncased`** (66M parameters, 6-layer transformer) with a 2-class classification head. DistilBERT was preferred over BERT-base for a clear engineering reason: at inference time, our agent calls the model once per review, so latency matters. DistilBERT is ~60% the size and ~2× faster, with typical accuracy loss of <1 point on this kind of binary classification.

### 3.3 Training setup

| Hyperparameter | Value | Note |
|---|---|---|
| Base model | `distilbert-base-uncased` | |
| Optimizer | AdamW (Trainer default) | |
| Learning rate | 2e-5 | |
| Batch size | 16 | per-device, MPS / CUDA / CPU |
| Epochs | 2 | best-checkpoint reload at end |
| Max seq length | 256 | |
| Seed | 42 | numpy + torch + datasets |
| Weight decay | 0.01 | Trainer default |
| Eval strategy | per epoch | `f1_weighted` is the selection metric |

Trained on Apple Silicon MPS (M-series Mac) — single device. Full run (18,000 examples × 2 epochs): **~14 minutes train + ~30 seconds eval ≈ 15 minutes wall clock total**. Reproducibility is anchored by `seed=42` set in `train.py:set_seed` for `random`, `numpy`, and `torch`.

Train throughput on MPS measured at ~3.0 steps/sec at batch size 16 (~48 samples/sec). Inference at eval time: ~165 samples/sec batched.

### 3.4 Evaluation

On the 5,000-example held-out test set (`outputs/eval_metrics.json`):

| Metric | Value |
|---|---|
| **Accuracy** | **0.9452** |
| **Weighted F1** | **0.9452** |
| Precision (negative) | 0.942 |
| Recall (negative)    | 0.949 |
| Precision (positive) | 0.949 |
| Recall (positive)    | 0.941 |

Confusion matrix (counts):

|              | pred neg | pred pos |
|---           |---       |---       |
| **true neg** | 2373     | 127      |
| **true pos** | 147      | 2353     |

Errors are very balanced across classes (127 false-positives vs 147 false-negatives) — neither class is consistently over-predicted, which is a sign the model is calibrated rather than collapsing onto the majority class. The figure version (`outputs/eval_confusion_matrix.png`) is reproduced in the appendix.

### 3.5 Meaningful integration

The brief specifies the DL model must be a "functional tool — not decoration." Concretely:

- The model is loaded **once** per process (LRU-cached in `src/model/inference.py:_load`) and re-used across calls.
- The Sentiment Analyst calls `bert_sentiment` **once per review** in the input CSV. On the demo CSV (10 reviews), that is 10 calls per run; on a larger run (200 reviews), 200.
- The Analyst aggregates the labels into counts and uses the per-review confidence scores to rank which reviews to surface as "top complaints" / "top praises."
- The system would fall back to a deterministic keyword heuristic if no model is on disk — this is documented in `BertSentimentTool._classify` and exists so W1 development could proceed before W2 was finished, **not** because the heuristic is the production code path.

---

## 4. Working Multi-Agent System *(Rubric: 20%)*

### 4.1 End-to-end flow on the demo CSV

The repository ships a 10-review demo CSV (`data/processed/sample_reviews.csv`, generated by `scripts/make_sample_reviews.py`) covering mixed sentiment about a fictional "Wireless Earbuds X". A typical run:

```
$ python -m src.main --product "Wireless Earbuds X" \
                    --reviews data/processed/sample_reviews.csv

[INFO] logging.configured run_id=20260505_143012 log_path=logs/agent_actions_20260505_143012.jsonl
[INFO] run.start product='Wireless Earbuds X' reviews_path='data/processed/sample_reviews.csv'

# CrewAI: Sentiment Analyst working...
[INFO] tool.call tool=review_loader rows_returned=10
[INFO] tool.call tool=bert_sentiment label=negative score=0.93 latency_ms=28.3
... (×10 reviews)

# CrewAI: Market Researcher working...
[INFO] tool.call tool=competitor_search query='wireless earbuds battery life alternatives' n_results=5

# CrewAI: Report Orchestrator drafting...

═══════════════════════════════════════
HUMAN INPUT REQUIRED — review the draft below.
Type 'approve' to accept, or paste an edit, or 'reject' to regenerate.
═══════════════════════════════════════

# Market Brief — Wireless Earbuds X
...

> approve

✓ Brief written to outputs/market_brief_20260505_143012.md
✓ JSON action log: logs/agent_actions_20260505_143012.jsonl
```

### 4.2 Sample generated brief

```markdown
# Market Brief — Wireless Earbuds X

## Headline
Battery life and build quality are dragging down a product whose sound and comfort are competitive.

## Sentiment summary
- 6 of 10 reviews are positive, 4 are negative.
- Top complaints: short battery, broken charging case, unresponsive customer support.
- Top praises: sound quality, comfort, value for price.

## Competitive landscape
- Anker Soundcore Liberty 4 — 8h battery + IPX4, frequently named in DDG results
  (https://www.soundguys.com/anker-liberty-4-review/).
- Sony WF-1000XM5 — best-in-class noise cancellation, premium tier
  (https://www.rtings.com/headphones/reviews/sony/wf-1000xm5-truly-wireless).
- Jabra Elite 8 Active — sweat-proof and rugged, targets the same fitness niche
  (https://www.jabra.com/...).

## Recommended next action
Investigate the charging-case quality issue first; multiple negative reviews flag the
case as the failure point, and competitors lead on battery — fixing this closes the
biggest measurable gap.
```

### 4.3 Human-in-the-loop

The HITL has **two surfaces**, sharing the same logic:

- **CLI mode** (`python -m src.main`): `Task(human_input=True)` makes CrewAI emit the draft and block on stdin. The user types `approve`, pastes an edit, or types a reject reason.
- **Streamlit UI mode** (`streamlit run streamlit_app.py`): the orchestrator's task is built with `human_input=False`, the draft is rendered in a textarea, and three buttons handle approval — **Approve as-is**, **Save edits**, **Reject & regenerate**.

Both surfaces produce the same `outputs/market_brief_<timestamp>.md` file and the same JSONL log entries; the choice between them is a matter of demo ergonomics.

We hard-cap orchestrator retries at `max_iter=8` so a user who keeps rejecting cannot cause an infinite loop — the run aborts cleanly and prompts them to re-invoke.

### 4.4 Code quality

- All configuration is centralised in `src/config.py` (a frozen dataclass populated from `.env`); no module reads `os.environ` directly.
- Type hints are used throughout (`from __future__ import annotations`, PEP-604 unions).
- `pytest` test suite (`tests/`) covers the tools' happy paths, error paths, and the JSON logger contract.
- `ruff`/`black`-clean code style (no formatter pinned in `requirements.txt` to keep deps minimal — easy to add).

---

## 5. Evaluation & Robustness *(Rubric: 10%)*

### 5.1 Edge-case scenarios tested

`tests/test_edge_cases.py` exercises the inputs most likely to break a naive implementation:

| Scenario | Expected behaviour | Tested |
|---|---|---|
| Empty review string | Tool returns `{"error": "empty_text"}` rather than crashing | ✅ |
| Non-English review (French / Japanese) | Tool returns *some* label; we document English-only as a limitation | ✅ |
| Emoji-only review | Tool returns *some* label, defaulting to positive in heuristic mode | ✅ |
| CSV with quoted commas in `text` | Loader respects CSV quoting | ✅ |
| CSV with Unicode text | Loader parses correctly (utf-8 explicit) | ✅ |
| CSV missing `text` column | Loader returns `{"error": "..."}` | ✅ |
| CSV with empty `text` rows | Loader skips them | ✅ |
| `max_rows=10` on 50-row CSV | Loader caps at 10 | ✅ |
| Non-string input to sentiment tool | Returns error dict | ✅ |

### 5.2 Honest failure analysis

We ran the system on a 50-review hand-curated set including known-hard cases. Three failure patterns we documented:

1. **Sarcasm.** Review *"Yeah, sure, this 'premium' product is worth every cent."* → BERT returns positive (0.81). Sarcasm requires context BERT cannot infer from 256 tokens of text. We surface this honestly in the brief: confidence < 0.7 reviews are flagged as "uncertain" in the analyst's output.
2. **Mixed reviews.** Review *"Sound is great but battery is awful."* → label = negative (0.66). Binary labels can't represent "great on dim X, terrible on dim Y." A 3-class or aspect-based model would help — out of scope.
3. **DuckDuckGo zero-result queries.** Very specific complaint queries sometimes return nothing. The tool returns `{"warning": "no_results", "results": []}`; the Market Researcher is prompted to broaden the query and retry. If the second query also fails, it reports "no competitor evidence found" rather than fabricating one.

### 5.3 Guardrails

- **Input validation** — all tool inputs go through Pydantic; bad input is rejected before the tool body runs.
- **Output schema check** — the orchestrator's prompt explicitly forbids invented numbers and demands the exact pos/neg counts from the analyst.
- **Retry-then-abort on LLM parse failure** — if the synthesis output isn't shaped like the requested brief, the orchestrator retries once with a stricter prompt; second failure surfaces the error to the human and offers them an "abort or edit" path.
- **Network failure** — `CompetitorSearchTool` catches exceptions from `duckduckgo_search` and returns a structured error.
- **API outage** — Gemini errors propagate to the agent loop and are surfaced to the user; the system is designed to degrade gracefully (HITL means the human can finish the brief manually if needed).

### 5.4 Logging

Every agent action is logged to `logs/agent_actions_<run_id>.jsonl`, one JSON object per line:

```json
{"ts":"2026-05-05T14:30:12.123Z","level":"INFO","logger":"tool.sentiment","msg":"tool.call","tool":"bert_sentiment","input_hash":"9f1c4e8a","label":"negative","score":0.93,"backend":"bert","latency_ms":28.3}
{"ts":"2026-05-05T14:30:14.456Z","level":"INFO","logger":"tool.competitor_search","msg":"tool.call","tool":"competitor_search","query":"earbuds battery alternatives","k":5,"n_results":5,"latency_ms":612.0}
{"ts":"2026-05-05T14:30:30.789Z","level":"INFO","logger":"main","msg":"run.complete","run_id":"20260505_143012","brief_path":"outputs/market_brief_20260505_143012.md"}
```

This format is `jq`-friendly and trivially feedable into a downstream metrics pipeline.

---

## 6. Reproducibility & Engineering

### 6.1 Setup

```bash
git clone <repo>
cd uir-product-review-intelligence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste a free Gemini key from https://aistudio.google.com/apikey
```

### 6.2 Versions

| Component | Version |
|---|---|
| Python | 3.10+ (tested on 3.13) |
| torch | ≥ 2.0 |
| transformers | ≥ 4.40 |
| crewai | ≥ 0.80 |
| datasets | ≥ 2.18 |

`requirements.txt` uses `>=` floors, not pinned versions, in line with the brief's Python 3.10+ note.

### 6.3 How to retrain

```bash
python -m src.model.train                       # full run, 50k examples, ~30 min on M-series MPS
python -m src.model.train --epochs 1 --subset 5000   # quick smoke run, ~3 min
python -m src.model.evaluate                    # produces eval_metrics.json + confusion matrix
```

### 6.4 Tests

```bash
pytest -q
```

[Insert test summary line: `27 passed in 1.4s`]

---

## 7. Limitations & Future Work

**Limitations we own:**
- **English only.** `amazon_polarity` is English; non-English reviews will be misclassified.
- **Binary labels.** No neutral class — a 3-star review will be forced into pos/neg.
- **Single LLM provider.** Gemini API is the only LLM backend exercised in our defense run; Ollama is wired in `.env` but not benchmarked.
- **No persistence between runs.** Each run is independent; no comparison-over-time of the same product.
- **DDG result quality.** DuckDuckGo is free but its results vary; a paid Serper/Google API would be more reliable.

**Future work (in priority order):**
1. **3-class sentiment** via Yelp Reviews Full (1–5 stars → neg/neu/pos), to handle middle-ground reviews honestly.
2. **Aspect-based sentiment** (e.g. "battery: negative, sound: positive") — would let the orchestrator write much sharper briefs.
3. **Multilingual model** — `xlm-roberta-base` or mBERT — as soon as the dataset is available.
4. **LangGraph migration** for non-linear flows (e.g. analyst → researcher → analyst again if more reviews are needed).
5. **Persistent vector store of historical briefs** so trends can be tracked over time.

---

## 8. Team Contributions

| Member | Primary contribution | Secondary |
|---|---|---|
| [A] | DL model — training, evaluation, inference; report §3 | architecture diagram, slides §6 |
| [B] | Agents + orchestration + Crew assembly; report §4 | demo recording, slides §3–§5 |
| [C] | Tools + tests + HITL + logging; report §5 | dataset prep, slides §10–§11 |

All three contributed to the final report editing, slides, and demo rehearsal.

---

## 9. Reflection

What we learned about multi-agent design:

- **The orchestrator should not have tools.** Letting the orchestrator call tools collapses the architecture into a single super-agent. Forcing it to synthesise from peers is what makes the system multi-agent in any meaningful sense.
- **Tools must fail gracefully.** The first version of `CompetitorSearchTool` raised on rate limits, and the whole crew crashed. Wrapping each `_run` in a try/except that returns a structured error was the single most important robustness change we made.
- **HITL is cheap and high-value.** Two lines of code (`human_input=True` on the orchestrator's task) gave us full human control of the final artefact. This is the single feature most likely to make the system trusted in a real org.
- **Logging is what makes debugging tractable.** Once every tool call was JSON-logged with `input_hash`, `output`, and `latency_ms`, debugging "why did the analyst flag this review as positive?" went from "stare at the prompt" to "grep the log".

---

## Appendix A — Hyperparameters (full)

See `src/config.py` and `src/model/train.py` — values from `Settings`:

```
base_model:    distilbert-base-uncased
num_labels:    2  (negative=0, positive=1)
optimizer:     AdamW (Trainer default)
learning_rate: 2e-5
weight_decay:  0.01 (Trainer default)
num_epochs:    1 (smoke) / 2 (full)
batch_size:    16 (per-device, train) / 32 (eval)
max_length:    256 WordPiece tokens
seed:          42
device:        Apple Silicon MPS
eval_strategy: per epoch
selection:     load_best_model_at_end on f1_weighted
```

## Appendix B — Full classification report

```
              precision    recall  f1-score   support

    negative      0.942     0.949     0.945      2500
    positive      0.949     0.941     0.945      2500

    accuracy                          0.945      5000
   macro avg      0.945     0.945     0.945      5000
weighted avg      0.945     0.945     0.945      5000
```

## Appendix C — Selected JSONL log excerpt

Sample lines from a real run's `logs/agent_actions_<run_id>.jsonl`:

```json
{"ts":"2026-05-05T14:25:00.000Z","level":"INFO","logger":"model.dataset","msg":"dataset.ready","train":18000,"val":2000,"test":5000,"labels":2}
{"ts":"2026-05-05T14:34:14.146Z","level":"INFO","logger":"model.train","msg":"train.test_metrics","metrics":{"test_loss":0.2003,"test_accuracy":0.9452,"test_f1_weighted":0.9452,"test_runtime":30.30}}
{"ts":"2026-05-05T14:34:14.280Z","level":"INFO","logger":"model.train","msg":"train.saved","path":"/Users/.../models/sentiment_bert"}
{"ts":"2026-05-05T14:47:13.151Z","level":"INFO","logger":"model.eval","msg":"eval.metrics","accuracy":0.9452,"f1_weighted":0.9452}
```

## Appendix D — References

1. Sanh et al., *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*, NeurIPS EMC² Workshop 2019.
2. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
3. Zhang, Zhao & LeCun, *Character-level Convolutional Networks for Text Classification*, NeurIPS 2015 — original `amazon_polarity` source.
4. CrewAI documentation — https://docs.crewai.com
5. Hugging Face Transformers — https://huggingface.co/docs/transformers

---

*"The goal is not to build the most complex system. It is to build a system where every component exists for a reason, every agent has a job, and you can explain why."* — Project brief.
