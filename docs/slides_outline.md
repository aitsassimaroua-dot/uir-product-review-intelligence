# Defense Slides — Outline (10–12 slides for 15-min defense + 10-min Q&A)

| # | Slide | What's on it | Speaker note |
|---|---|---|---|
| 1 | Title | Project name, team, course, prof, date | Hook in 1 sentence: "We turn 200 product reviews into a 1-page brief in 90 seconds." |
| 2 | Problem | 1 image (PM scrolling reviews) + 3 pain points | Why automation is needed |
| 3 | Our system in 1 picture | The mermaid architecture diagram (PNG) | 30 seconds — point at each box |
| 4 | Why multi-agent (not one prompt) | 4-row table: problem → our fix | This is the design-rationale slide; rubric line: "agent roles justified" |
| 5 | Agent 1 — Sentiment Analyst + BERT | Role · tool · I/O example | Show 1 actual JSON tool-call from the log |
| 6 | The DL model | Dataset → DistilBERT → metrics card | Headline: accuracy + F1 + confusion matrix |
| 7 | Agent 2 — Market Researcher | Role · tool (DDG) · sample query | Quick example output |
| 8 | Agent 3 — Orchestrator + HITL | Code snippet: `human_input=True` + screenshot of prompt | This is the rubric line: "HITL present" |
| 9 | Live demo *(or video fallback)* | Run on `data/processed/sample_reviews.csv` | If demo gods are angry: 30s recorded clip |
| 10 | Robustness | Edge-case table + 1 honest failure example | Shows we tested it |
| 11 | Logging + reproducibility | 1 JSONL line + `pip install + .env + make demo` | Rubric line: "logging" + "reproducibility" |
| 12 | What we'd change | 3 bullets: limitations · future work · learning | Don't oversell — graders like honesty |

## Defense Q&A — likely questions, prepared answers

| Likely question | Our answer |
|---|---|
| Why DistilBERT and not BERT-base? | DistilBERT trains in ~half the time on free Colab with ~1% F1 loss. Tradeoff documented in report §3. |
| Why CrewAI and not LangGraph? | Brief specifies CrewAI as primary stack; LangGraph is the stretch goal. CrewAI's `human_input=True` gives HITL out of the box. |
| Why 3 agents and not 2 or 5? | Smallest count that justifies the design. 2 = no orchestration value; 5 = coordination cost without new capability. |
| What happens if Gemini API is down? | Tool-level try/except returns structured error; orchestrator surfaces it to the user; HITL lets the user retry. We don't auto-fail-over to Ollama in current scope but it's wired in `.env`. |
| Why amazon_polarity? | Direct domain match (product reviews); 3.6M source = plenty of headroom for stratified subsetting. Documented in `docs/dataset_choice.md`. |
| Where's the human-in-the-loop? | `Task(human_input=True)` in `src/crew.py:write_brief` — terminal prompt before final brief is written. |
| How do you log every agent action? | JSON formatter at `src/utils/logging_config.py`; one JSONL file per run in `logs/`; tools log on each `_run`. |
| Can the system make up numbers? | Orchestrator prompt explicitly forbids invented figures; it can only quote analyst output. Guardrail not foolproof — covered in honest failure analysis. |
| What dataset bias did you find? | 3-star reviews excluded by source = no neutral class; English only; reviews skew older (~2015). Documented. |
| Why not a web UI? | Brief asks for working multi-agent + HITL, not UX. Terminal HITL satisfies the requirement at lower complexity. |

## Demo script (for the 90-second live run)

1. `cat data/processed/sample_reviews.csv | head -3` — show input.
2. `python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv`
3. Narrate: "Sentiment Analyst classifies each review with BERT — note the JSON log appearing in `logs/`."
4. "Now Market Researcher searches DuckDuckGo for competitors."
5. "Orchestrator drafts the brief — and *here's the HITL prompt*. I'll approve."
6. `cat outputs/market_brief_*.md | head -30` — show output.
7. `tail -3 logs/agent_actions_*.jsonl | python -m json.tool` — prove logs are valid JSON.
