# Product Review Intelligence — Multi-Agent AI System

> S8 Integrated Project — UIR | AI & Big Data Program | 2025–2026
> Prof. Hakim Hafidi

Multi-agent system that analyzes product reviews, scouts competitors, and produces a market intelligence brief. A fine-tuned BERT classifier provides the sentiment analysis backbone, wrapped as a CrewAI tool used by a specialized agent.

## Why this design

Three agents collaborate, each with a clear responsibility:

| Agent | Role | Tools |
|---|---|---|
| **Sentiment Analyst** | Classifies reviews + extracts pain points | `BertSentimentTool` (our DL model), `ReviewLoaderTool` |
| **Market Researcher** | Scouts competitor reviews + market context | `CompetitorSearchTool` |
| **Report Orchestrator** | Coordinates, asks human approval, writes brief | (synthesizes outputs of the other two) |

A **human-in-the-loop checkpoint** sits before the orchestrator finalizes the market brief: the user reviews/edits the synthesis before export.

See `docs/architecture.md` for the full rationale, communication diagram, and design trade-offs.

## Tech stack

- **Python** 3.10+
- **Agent framework**: CrewAI
- **LLM backend**: Gemini 1.5 Flash (free API) — switchable to Ollama
- **DL framework**: PyTorch ≥ 2.0 + Hugging Face `transformers`
- **Dataset**: `amazon_polarity` (subset, see `docs/dataset_choice.md`)

## Setup

### 1. Clone and install

```bash
git clone <this-repo>
cd uir-product-review-intelligence
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Create a key (free tier, no card required)
3. Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```

### 3. Train the BERT model (W2)

```bash
python -m src.model.train --epochs 2 --subset 50000
```

This downloads `amazon_polarity`, fine-tunes `distilbert-base-uncased`, saves to `models/sentiment_bert/`, and writes evaluation metrics + confusion matrix to `outputs/`.

### 4. Run the multi-agent system (W3)

**Option A — CLI** (terminal HITL):
```bash
python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv
```

**Option B — Streamlit UI** (recommended for the demo):
```bash
streamlit run streamlit_app.py
# or: make ui
```
Opens at http://localhost:8501 — upload/paste reviews, run agents, approve/edit/reject the draft via buttons, download the final brief and JSON log.

In both modes the system will:
1. Load reviews → Sentiment Analyst classifies each one
2. Market Researcher fetches competitor context
3. Orchestrator drafts a market brief
4. **HITL checkpoint**: you review the draft → approve / edit / reject
5. Final brief written to `outputs/market_brief_<timestamp>.md`

Every agent action is logged as JSON to `logs/agent_actions_<timestamp>.jsonl`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── docs/
│   ├── architecture.md          # Agent design + diagrams + rationale
│   ├── dataset_choice.md        # Why amazon_polarity, preprocessing
│   └── timeline.md              # W1–W4 plan
├── src/
│   ├── config.py                # Centralized settings (env-driven)
│   ├── utils/logging_config.py  # JSON structured logging
│   ├── model/                   # BERT: dataset, training, eval, inference
│   ├── tools/                   # CrewAI tools (sentiment, search, loader)
│   ├── agents/                  # 3 agents (analyst, researcher, orchestrator)
│   ├── crew.py                  # Crew assembly + orchestration
│   └── main.py                  # CLI entry point with HITL
├── tests/                       # Unit tests (tools, logging, eval)
├── scripts/                     # Helper scripts (download_data, demo)
├── data/                        # Raw + processed reviews (gitignored)
├── models/                      # Trained checkpoints (gitignored)
├── logs/                        # JSONL action logs (gitignored)
└── outputs/                     # Generated briefs + figures
```

## Deliverables checklist

- [x] **W1** — Single-agent prototype, architecture doc, domain + dataset chosen
- [ ] **W2** — Trained BERT with eval (accuracy, F1, confusion matrix), wrapped as tool
- [ ] **W3** — 2 specialists + orchestrator, HITL checkpoint, end-to-end demo
- [ ] **W4** — Guardrails, edge-case tests, 8–12 page report, demo video, slides

## Evaluation criteria coverage (per brief)

| Criterion | Where it lives |
|---|---|
| Agent roles justified, communication diagram | `docs/architecture.md` |
| Trained DL model with rigorous evaluation | `src/model/`, `outputs/eval_*.png` |
| 2+ tools with I/O schemas | `src/tools/` (each tool has Pydantic schemas) |
| Human-in-the-loop checkpoint | `src/main.py` (CrewAI `human_input=True`) |
| Error handling, no crashes | `try/except` in every tool, `tests/` |
| JSON action logs with timestamps | `src/utils/logging_config.py` → `logs/*.jsonl` |
| Reproducibility | this README + `requirements.txt` + `.env.example` |

## Team

3 students — UIR S8, AI & Big Data Program.

## License

Academic project — UIR 2025–2026.
