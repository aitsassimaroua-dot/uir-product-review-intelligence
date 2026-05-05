# Final Handoff — What's Done, What You Must Do

> Read this first. Tells you exactly what's ready and what only **you** can do.

## ✅ What's already done (in this repo)

- [x] **Code complete** — 27 Python files, ~1200 lines. Every Python file parses cleanly.
- [x] **3 agents** built and wired: Sentiment Analyst, Market Researcher, Report Orchestrator (`src/agents/`).
- [x] **3 tools** with Pydantic I/O schemas: `BertSentimentTool`, `ReviewLoaderTool`, `CompetitorSearchTool` (`src/tools/`).
- [x] **BERT pipeline** ready to train: `src/model/{dataset,train,evaluate,inference}.py`.
- [x] **Multi-agent orchestration** with HITL: `src/crew.py`, `src/main.py`.
- [x] **JSON structured logging** — every agent action logged to `logs/agent_actions_<run_id>.jsonl`.
- [x] **Test suite** (`tests/`) — 5 files covering happy paths, error paths, edge cases.
- [x] **All documentation**:
  - `docs/architecture.md` — architecture + diagram + rationale (rubric: System Design 15%)
  - `docs/dataset_choice.md` — why amazon_polarity, preprocessing, honest caveats
  - `docs/timeline.md` — W1→W4 plan
  - `docs/report.md` — **full 8–12 page report**, ready to render to PDF
  - `docs/slides.md` — **full 12-slide deck**, Marp-ready
  - `docs/demo_script.md` — beat-sheet for the 3-5 min demo video
  - `docs/slides_outline.md` — Q&A prep for the defense
- [x] **Dev environment** — `.venv/` with all dependencies installed.
- [x] **Sample reviews CSV** for the demo (`data/processed/sample_reviews.csv`).
- [x] **Configuration** — `.env.example`, `requirements.txt`, `Makefile`, `.gitignore`.

## ⚠️ What only YOU can do

### 1. Get your free Gemini API key (5 min)
1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account → "Create API key"
3. Copy the key
4. `cp .env.example .env`, then edit `.env` and paste:
   ```
   GEMINI_API_KEY=AIza…your_key…
   ```

Without this, the agents can't call any LLM.

### 2. Train the BERT model (~30 min on M-series Mac)
```bash
source .venv/bin/activate
python -m src.model.train
```
This downloads `amazon_polarity` (~250 MB), trains for 2 epochs on a 50k subset, saves the model to `models/sentiment_bert/`, and writes `test_metrics.json`.

For a quick smoke test (~3 min): `python -m src.model.train --epochs 1 --subset 5000`.

### 3. Run evaluation
```bash
python -m src.model.evaluate
```
Produces:
- `outputs/eval_metrics.json` — accuracy, F1, confusion matrix as numbers
- `outputs/eval_classification_report.txt` — per-class precision/recall/F1
- `outputs/eval_confusion_matrix.png` — figure for the report

### 4. Run the multi-agent demo
```bash
python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv
```
You'll see the JSON log fill up, agents take turns, and a HITL prompt before the brief is finalized. **Type `approve`** to accept the draft.

The brief is written to `outputs/market_brief_<timestamp>.md`.

### 5. Fill in the placeholders in `docs/report.md`
After training + eval, edit `docs/report.md` and replace these markers with real values:
- `[ACC]` → accuracy from `outputs/eval_metrics.json`
- `[F1]` → weighted F1
- `[P_neg]`, `[R_neg]`, `[P_pos]`, `[R_pos]` → from the classification report
- `[T]` → minutes the training took (look at the wall clock)
- `[Member A/B/C]` → real names + UIR emails
- Appendix B/C — paste actual file contents
- Demo brief sample (§4.2) — replace with a real generated brief

### 6. Convert `docs/report.md` to PDF
Easiest path:
```bash
# Option A: pandoc
brew install pandoc basictex
pandoc docs/report.md -o docs/report.pdf --toc --pdf-engine=xelatex \
       -V geometry:margin=2.5cm -V documentclass=article

# Option B: Markdown → HTML → print to PDF in a browser
# Open docs/report.md in VSCode/Cursor, use the "Markdown PDF" extension.
```
Target: 8–12 pages. Current draft is sized for ~12 — trim if needed.

### 7. Convert `docs/slides.md` to PDF
```bash
npm install -g @marp-team/marp-cli
marp docs/slides.md --pdf -o docs/slides.pdf
# or:
marp docs/slides.md --pptx -o docs/slides.pptx     # PowerPoint
```

### 8. Record the 3–5 min demo video
1. Read `docs/demo_script.md` once.
2. Open QuickTime / OBS / Loom, screen-record.
3. Run the demo command live.
4. Voice-over per the script's beat sheet.
5. Save as `docs/demo_video.mp4`.

### 9. Push to GitHub
```bash
# Create a repo on github.com (private or public)
git remote add origin https://github.com/<you>/uir-product-review-intelligence.git
git branch -M main
git push -u origin main
```
Update `docs/report.md` and `docs/slides.md` with the real repo URL.

### 10. Defense rehearsal (1 hour)
1. Read `docs/slides_outline.md` "Q&A — likely questions" section.
2. Time yourself going through the slides — target **15 minutes**.
3. Practice answering the Q&A prep questions out loud.
4. Make sure you can explain *every line* of code in `src/agents/`, `src/tools/`, `src/crew.py`, and `src/model/inference.py` (the prof said you must defend every line).

## Estimated time to finish (your end)

| Step | Time |
|---|---|
| Gemini key setup | 5 min |
| BERT training (full) | ~30 min on Mac MPS, runs unattended |
| Evaluation | 2 min |
| Demo run + brief generation | 3 min |
| Fill placeholders in report | 30 min |
| Render report to PDF | 10 min |
| Render slides to PDF | 5 min |
| Record demo video (3 takes) | 30 min |
| Push to GitHub | 5 min |
| Defense rehearsal | 60 min |
| **Total** | **~2.5 hours of your active time** + 30 min training waiting |

## Submission checklist (per brief)

- [ ] GitHub repo URL
- [ ] Trained model — uploaded as Git LFS *or* shared via Drive link in README (the model is ~270 MB; check repo size limits)
- [ ] PDF report (8–12 pages) — `docs/report.pdf`
- [ ] Demo video (3–5 min) — `docs/demo_video.mp4` (or YouTube unlisted)
- [ ] Slides — `docs/slides.pdf`
- [ ] Final deadline: end of W4

## If something goes wrong

| Symptom | Likely fix |
|---|---|
| `pip install` fails on `crewai` | Check Python version; use 3.10/3.11 if 3.13 is too new |
| `GEMINI_API_KEY is empty` | Did you `cp .env.example .env` and fill the key? |
| `No model found at models/sentiment_bert` when running demo | Run `python -m src.model.train` first — the agent falls back to a stub heuristic if no model is on disk |
| DuckDuckGo returns nothing | Free DDG sometimes rate-limits. Wait a minute, try again, or change the search query |
| Training is too slow on your Mac | Run with `--subset 10000 --epochs 1` for a much faster run; document the smaller subset in the report |
| Out of memory during training | Drop `BATCH_SIZE=8` in `.env` |

## File you must not commit

- `.env` (your real Gemini key)
- `models/sentiment_bert/` (too big for free GitHub — share via Drive link)
- `data/raw/` (datasets are downloadable from HF, no need to commit)

All three are already gitignored.

---

Good luck on the defense.
