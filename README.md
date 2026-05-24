# Product Review Intelligence

> A multi-agent AI system for customer insight and competitive analysis.
> S8 Integrated Project · UIR · Big Data & AI · 2025–2026

A three-agent system that classifies product reviews with a fine-tuned DistilBERT model,
scouts competitors via web search, and produces a one-page market brief — with a
human-in-the-loop checkpoint before publication.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aitsassimaroua-dot/uir-product-review-intelligence/blob/main/notebooks/demo.ipynb)

## The team

- **Chaymae Benmakhlouf** · **Maroua Ait Sassi** · **Rania Bouaroua**
- Supervised by **Prof. Hakim Hafidi**

## Deliverables

| Item | Location |
|---|---|
| Demo notebook (Colab-ready) | [`notebooks/demo.ipynb`](notebooks/demo.ipynb) |
| Written report | [`deliverables/REPORT.pdf`](deliverables/REPORT.pdf) |
| Defense presentation | [`deliverables/PRESENTATION.pptx`](deliverables/PRESENTATION.pptx) |
| Demo video | [youtu.be/3ngsXPVdwVw](https://youtu.be/3ngsXPVdwVw) |

## The three agents

| Agent | Role | Tools |
|---|---|---|
| **Sentiment Analyst** | Classifies each review, extracts top complaints and praises | `bert_sentiment`, `review_loader` |
| **Market Researcher** | Uses the analyst's complaints to scout three plausible competitors | `competitor_search` |
| **Report Orchestrator** | Synthesises a one-page brief, then waits for human approval | _(synthesis only)_ |

A communication diagram and the full rationale live in [`docs/architecture.md`](docs/architecture.md).

## Quick start (locally)

```bash
git clone https://github.com/aitsassimaroua-dot/uir-product-review-intelligence.git
cd uir-product-review-intelligence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste a free Gemini key from https://aistudio.google.com/apikey
python scripts/make_sample_reviews.py    # generate the 10-review demo CSV
```

### Train the BERT classifier

```bash
python -m src.model.train                       # full run: 50k examples, 2 epochs (~30 min on MPS)
python -m src.model.train --epochs 1 --subset 5000   # quick smoke run (~3 min)
python -m src.model.evaluate                    # writes outputs/eval_metrics.json + confusion matrix
```

### Run the multi-agent pipeline

**CLI (terminal HITL):**
```bash
python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv
```

**Streamlit UI (recommended for the demo):**
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501 — upload/paste reviews, run agents, approve/edit/reject the draft via buttons.

## Reproducibility

- All settings centralised in `src/config.py`, driven by `.env`. No `os.environ` calls elsewhere.
- Random seed `42` set for `random`, `numpy`, `torch`, and Hugging Face datasets.
- Every agent action is appended to `logs/agent_actions_<run_id>.jsonl` with input hash, output, and latency.
- 20 unit tests in `tests/` run in ~11 seconds: `pytest -q`.

## Project structure

```
.
├── README.md                  # this file
├── requirements.txt
├── .env.example
├── notebooks/
│   └── demo.ipynb            # Colab-ready end-to-end demo
├── src/
│   ├── config.py             # centralised settings (env-driven)
│   ├── model/                # DistilBERT: dataset, training, eval, inference
│   ├── tools/                # CrewAI tools (sentiment, search, loader)
│   ├── agents/               # 3 agents (analyst, researcher, orchestrator)
│   ├── crew.py               # crew assembly + orchestration
│   ├── main.py               # CLI entry point with HITL
│   └── utils/                # JSONL logging
├── streamlit_app.py          # browser UI (alternative to CLI)
├── scripts/
│   └── make_sample_reviews.py  # generate the 10-review demo CSV
├── tests/                    # pytest suite
├── deliverables/             # report (.pdf), slides (.pptx)
├── docs/                     # architecture, dataset choice, etc.
├── outputs/                  # eval metrics, generated briefs (gitignored)
└── logs/                     # JSONL action logs (gitignored)
```

## Stack

- **Python** 3.10+
- **Agents** — [CrewAI](https://docs.crewai.com)
- **LLM** — Gemini 1.5 / 2.0 Flash (free tier) · switchable to Ollama
- **Deep learning** — PyTorch ≥ 2.0 + Hugging Face `transformers`
- **Dataset** — `amazon_polarity` (see [`docs/dataset_choice.md`](docs/dataset_choice.md))
- **UI** — Streamlit

## Headline numbers

| Metric | Value |
|---|---|
| Test accuracy (5,000 held-out reviews) | **0.9452** |
| Weighted F1 | **0.9452** |
| End-to-end run on 10 reviews | **≈ 90 s** |
| Unit tests | **20 / 20 pass** |

## License

Academic project — UIR 2025–2026.
