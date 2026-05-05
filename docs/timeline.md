# 4-Week Plan — Product Review Intelligence

## W1 — Agent Foundations & Scoping
**Goal**: Working single-agent prototype, architecture doc, domain + dataset chosen.

- [x] Project scaffold (`src/`, `docs/`, `tests/`, `.env.example`)
- [x] `docs/architecture.md` with mermaid diagram + role rationale
- [x] `docs/dataset_choice.md`
- [x] JSON structured logging (`src/utils/logging_config.py`)
- [x] Single-agent prototype with placeholder sentiment tool (`src/agents/sentiment_analyst.py` + `src/tools/sentiment_tool.py` returning a heuristic until W2 model lands)
- [x] CrewAI ↔ Gemini wiring tested
- [x] README setup instructions

## W2 — DL Model as Agent Tool
**Goal**: Trained BERT model + evaluation; model wrapped as a tool the agent actually calls.

- [ ] `scripts/download_data.py` — pulls `amazon_polarity`, stratified subset → `data/processed/`
- [ ] `src/model/dataset.py` — `torch.utils.data.Dataset` wrapper with tokenization
- [ ] `src/model/bert_classifier.py` — `AutoModelForSequenceClassification` head
- [ ] `src/model/train.py` — Hugging Face `Trainer` loop, seed=42, saves checkpoint
- [ ] `src/model/evaluate.py` — accuracy, precision/recall/F1 per class, confusion matrix → PNG in `outputs/`
- [ ] `src/model/inference.py` — single-shot + batched API, used by tools
- [ ] Replace placeholder `BertSentimentTool` with real model
- [ ] Training run report: hyperparams, loss curves, eval table → `docs/training_report.md`

**Target metrics** (from prior work on this dataset): accuracy ≥ 0.93, F1 ≥ 0.92.

## W3 — Multi-Agent Orchestration
**Goal**: 2 specialists + orchestrator, HITL, end-to-end demo.

- [ ] `src/agents/market_researcher.py` + `src/tools/competitor_search_tool.py` (DuckDuckGo)
- [ ] `src/agents/orchestrator.py`
- [ ] `src/crew.py` — Crew assembly, sequential process, task chain
- [ ] `src/main.py` — CLI with `--product` and `--reviews` flags, HITL prompt loop
- [ ] Sample reviews CSV in `data/processed/sample_reviews.csv` for demo
- [ ] End-to-end run produces `outputs/market_brief_<ts>.md`
- [ ] All three agents log JSON entries

## W4 — Evaluation & Defense
**Goal**: Guardrails, edge cases, final report, slides, demo video.

- [ ] `tests/` — unit tests for tools (input validation, error paths)
- [ ] Edge-case scenarios scripted in `tests/test_edge_cases.py`:
  - empty reviews list
  - non-English text
  - sarcastic/ironic reviews from a hand-picked set
  - DuckDuckGo returns 0 results
  - Gemini API outage (mocked) → graceful degradation
- [ ] Guardrails — input sanitization, output JSON schema validation
- [ ] `docs/report.pdf` — 8–12 pages, structure in `docs/report_skeleton.md`
- [ ] `docs/slides.pdf` — 10–12 slides
- [ ] Demo video (3–5 min) — script in `docs/demo_script.md`
- [ ] Final defense rehearsal

## Roles in the team (3 students)

Suggested split — adjust after team discussion:

| Member | Primary | Secondary |
|---|---|---|
| A | DL model (W2) — training, eval, inference | report ML section |
| B | Agents + orchestration (W3) | report architecture section |
| C | Tools + tests + HITL (W3, W4) | report eval/edge-cases section |

All three contribute to slides and demo video.
