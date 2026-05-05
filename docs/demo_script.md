# Demo Video Script — 3 to 5 minutes

> One take, screen recorded with voiceover. Keep it tight.

## Setup before recording
- Terminal at full screen, font 16+, dark theme.
- `.env` already filled with valid Gemini key.
- Model trained (`models/sentiment_bert/config.json` exists).
- Sample reviews ready: `python scripts/make_sample_reviews.py`.
- Nothing else running on the laptop (close Slack, etc.).

## Beat sheet (~4 minutes)

| Time | Beat | What's on screen | Voiceover (script) |
|---|---|---|---|
| 0:00–0:15 | Hook | Title slide | "We built a multi-agent AI system that turns 200 product reviews into a 1-page market brief in under two minutes." |
| 0:15–0:45 | Problem | Quick screen of an Amazon product page with 1000+ reviews | "A product manager who wants signal from these reviews currently has to scroll for hours. Our system does it for them." |
| 0:45–1:15 | Architecture | `docs/architecture.md` mermaid diagram | "Three agents: a Sentiment Analyst that owns a fine-tuned BERT, a Market Researcher that scouts competitors, and an Orchestrator that synthesizes — with a human checkpoint before the brief is finalized." |
| 1:15–1:30 | Show input | `cat data/processed/sample_reviews.csv` | "Here are 10 reviews — mixed sentiment, real product complaints." |
| 1:30–2:30 | Live run | `python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv` | "Watch the JSON log fill up — every agent action is timestamped. Sentiment Analyst calls BERT 10 times. Then the Market Researcher searches the web. Now the Orchestrator drafts the brief — and here's the human checkpoint. I review the draft, approve." |
| 2:30–3:00 | Show output | `cat outputs/market_brief_*.md` | "One-page brief: headline, sentiment summary with exact counts, three competitors with URLs, recommended action." |
| 3:00–3:30 | Show logs | `tail -5 logs/agent_actions_*.jsonl \| python -m json.tool` | "Every action is logged as structured JSON. Auditable, replayable, debuggable." |
| 3:30–4:00 | Wrap | Architecture slide back on | "BERT fine-tuned on amazon_polarity at 0.94 accuracy. Three agents, two tools, one human checkpoint. Code on GitHub. Thanks." |

## Re-record triggers
- LLM call > 30 s — cut and re-do (Gemini sometimes throttles).
- Any traceback — fix and re-do.
- Filler words ("um", "uh") — re-take that segment.

## Export
- 1080p, 30 fps, MP4 H.264, < 50 MB ideally.
- Filename: `demo_product_review_intelligence.mp4`.
