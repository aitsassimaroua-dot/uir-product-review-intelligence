# Final Report — Skeleton (8–12 pages)

> Use this as the table-of-contents for the final PDF. Each section maps to a rubric criterion.

---

## Page 1 — Cover
- Title: **Product Review Intelligence — Multi-Agent AI System**
- Course: S8 Integrated Project — UIR — AI & Big Data Program — 2025–2026
- Team: 3 names + emails
- Date, professor: Hakim Hafidi
- Repo URL + demo video URL

## Page 2 — Executive Summary (½ page)
- Problem (2 sentences): manual review-scrolling is slow; PMs need a faster brief.
- Our system (2 sentences): 3 specialized agents + fine-tuned DistilBERT + HITL.
- Result headline: BERT accuracy / F1, end-to-end latency for 10 reviews, key qualitative finding from a demo run.

## Page 2-3 — System Design & Architecture *(rubric: 15%)*
- Why multi-agent (table from `docs/architecture.md`).
- 3 agents: roles, tools, why specialized.
- Communication diagram (mermaid → exported as PNG).
- Sequential process + HITL placement rationale.
- Trade-offs we considered (3 agents vs 5, DistilBERT vs BERT, DDG vs Serper, CrewAI vs LangGraph).

## Page 3-5 — DL Model Integration *(rubric: 15%)*
- Dataset: `amazon_polarity`, why, subsetting strategy, 50k stratified.
- Preprocessing: title+content concat, max_length=256.
- Model: `distilbert-base-uncased`, head, why DistilBERT (training time table).
- Training: hyperparameters table (lr, batch, epochs, seed=42), loss curves figure.
- **Evaluation** (rigorous): accuracy, weighted F1, per-class precision/recall, confusion matrix figure, classification report.
- Inference latency: ms/review, batched throughput.
- **Meaningful integration**: how the agent calls it (1 review → 1 call, batched per task), what the agent does with the output (aggregates counts, surfaces complaints).

## Page 5-7 — Working Multi-Agent System *(rubric: 20%)*
- End-to-end flow on the demo CSV — annotated screenshots of the run.
- HITL screenshot — terminal prompt + user edit shown.
- Tools: I/O schemas, error paths.
- Code-quality notes: typed configs, JSON logs, tests.
- A real generated `outputs/market_brief_*.md` reproduced inline as evidence.

## Page 7-8 — Evaluation & Robustness *(rubric: 10%)*
- BERT eval already covered above. This section = system-level eval.
- Edge cases (table from `tests/test_edge_cases.py`): empty input, non-English, sarcasm, unicode CSV, no search results, LLM API failure.
- Honest failure analysis: 3 examples where the system was wrong/unhelpful + why.
- Guardrails: input validation, output schema check, retry-then-abort on LLM parse failure, network error handling.
- Logging snippet (1 example JSONL line per agent).

## Page 8-9 — Reproducibility & Engineering
- Setup instructions (already in README — reference, don't duplicate).
- Versions table (Python, torch, transformers, crewai).
- How to retrain in <30 min on free Colab.
- Test results: `pytest -q` output.

## Page 9-10 — Demo, Limitations, Future Work
- Demo screenshots / link to video.
- Limitations: English only, binary labels (no neutral), small competitor list, single LLM provider.
- Future work: 3-class via Yelp Full, multilingual mBERT, LangGraph for DAG flows, vector store of historical briefs.

## Page 10-11 — Team Contributions & Reflection
- Who did what (link to git log if needed).
- What we learned about multi-agent design.
- What we'd change if we did it again.

## Page 11-12 — Appendix
- Full hyperparameters table.
- Full classification report (text).
- Selected JSONL log excerpt.
- References (CrewAI docs, HF transformers, papers cited).

---

## Figures to produce (and where they live)

| Figure | Source | Output path |
|---|---|---|
| Architecture diagram | mermaid in `docs/architecture.md` | `outputs/figs/architecture.png` |
| Confusion matrix | `src/model/evaluate.py` | `outputs/eval_confusion_matrix.png` |
| Loss curves | export from training (TensorBoard or matplotlib in train.py) | `outputs/figs/loss_curve.png` |
| Demo screenshots | manual capture of `make demo` | `outputs/figs/demo_*.png` |

## Tables to produce

- Hyperparameters
- Per-class precision/recall
- Edge-case scenarios + outcomes
- DistilBERT vs BERT-base time/accuracy comparison

## Page-budget sanity check

Cover (1) + Exec (0.5) + Design (2) + DL (3) + System (2) + Eval (1.5) + Repro (0.5) + Demo (1) + Team (0.5) + Appendix (1) ≈ **13 pages** — trim to fit 12.
