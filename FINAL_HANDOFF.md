# Final Handoff — What's Done, What You Must Do

> Read this first. Tells you exactly what's ready and what only **you** can do.

## ✅ What's already done (in this repo)

- [x] **Code complete** — 27 Python files, ~1200 lines. Every file parses, all 20 unit tests pass (`pytest -q`).
- [x] **3 agents** built and wired: Sentiment Analyst, Market Researcher, Report Orchestrator (`src/agents/`).
- [x] **3 tools** with Pydantic I/O schemas: `BertSentimentTool`, `ReviewLoaderTool`, `CompetitorSearchTool` (`src/tools/`).
- [x] **BERT trained** ✅ — `models/sentiment_bert/`, **94.52% accuracy / 0.9452 F1** on 5,000 held-out test reviews.
- [x] **Evaluation done** — `outputs/eval_metrics.json`, `eval_classification_report.txt`, `eval_confusion_matrix.png`.
- [x] **Multi-agent orchestration** with HITL: `src/crew.py`, `src/main.py`.
- [x] **JSON structured logging** — every agent action logged to `logs/agent_actions_<run_id>.jsonl`.
- [x] **Test suite** (`tests/`) — 5 files, **20 tests passing**, edge cases included.
- [x] **All documentation**:
  - `docs/architecture.md` — architecture + diagram + rationale (rubric: System Design 15%)
  - `docs/dataset_choice.md` — why amazon_polarity, preprocessing, honest caveats
  - `docs/timeline.md` — W1→W4 plan
  - `docs/report.md` — **full 8–12 page report with real numbers filled in**, ready to render to PDF
  - `docs/slides.md` — **full 12-slide deck with real metrics**, Marp-ready
  - `docs/demo_script.md` — beat-sheet for the 3-5 min demo video
  - `docs/slides_outline.md` — Q&A prep for the defense
- [x] **Dev environment** — `.venv/` with all dependencies installed (crewai 1.14, torch 2.11, transformers 5.7, etc.).
- [x] **Sample reviews CSV** for the demo (`data/processed/sample_reviews.csv`, 10 reviews mixed sentiment).
- [x] **Configuration** — `.env.example`, `requirements.txt`, `Makefile`, `.gitignore`.
- [x] **Git** — local repo initialized, 4 commits on `main`. Identity set to `elazouzi.khalil.ke@gmail.com`.

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

### 2. ~~Train the BERT model~~ — already done ✅
The model is at `models/sentiment_bert/` (~268 MB). Skip this step unless you want to retrain.

If you do want to retrain (e.g. with different hyperparameters):
```bash
source .venv/bin/activate
python -m src.model.train --epochs 2 --subset 20000
```

### 3. ~~Run evaluation~~ — already done ✅
Numbers and figures are in `outputs/`. Skip unless you want to re-evaluate.

### 4. Run the multi-agent demo (the only thing that requires your Gemini key)
```bash
source .venv/bin/activate
python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv
```
You'll see the JSON log fill up, agents take turns, and a HITL prompt before the brief is finalized. **Type `approve`** to accept the draft.

The brief is written to `outputs/market_brief_<timestamp>.md`.

Run this **at least once** before the defense — you need a real `outputs/market_brief_*.md` to paste into your report's §4.2.

### 5. Fill in the remaining placeholders in `docs/report.md`
The numbers are already filled in. Only these are still placeholders:
- `[Member A]` / `[Member B]` / `[Member C]` → your team's real names + UIR emails (cover, §8 Team Contributions)
- §4.2 sample brief — replace the example with whatever your real demo produces
- Cover-page repo + demo-video URLs (last lines of the cover and Page 1)

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

| Step | Time | Status |
|---|---|---|
| ~~BERT training~~ | ~~30 min~~ | ✅ done — 94.52% / 0.945 F1 |
| ~~Evaluation~~ | ~~2 min~~ | ✅ done |
| ~~Fill report numbers~~ | ~~30 min~~ | ✅ done |
| Gemini key setup | 5 min | TODO |
| Demo run + brief generation | 3 min | TODO |
| Replace 3 names + 1 sample brief in report | 5 min | TODO |
| Render report to PDF | 10 min | TODO |
| Render slides to PDF | 5 min | TODO |
| Record demo video (3 takes) | 30 min | TODO |
| Push to GitHub | 5 min | TODO |
| Defense rehearsal | 60 min | TODO |
| **Total remaining** | **~2 hours of your active time** | |

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
