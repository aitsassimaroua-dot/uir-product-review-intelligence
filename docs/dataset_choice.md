# Dataset Choice — `amazon_polarity`

## Why this dataset

| Criterion | `amazon_polarity` | IMDB | Yelp Polarity |
|---|---|---|---|
| Domain match (product reviews) | ✅ direct | ❌ movies | ⚠️ services/restaurants |
| Size | 3.6M train / 400k test | 25k / 25k | 560k / 38k |
| Labels | 2 (neg/pos) | 2 | 2 |
| HF availability | ✅ `amazon_polarity` | ✅ | ✅ |
| Avg review length | ~75 words | ~230 words | ~135 words |

We pick **`amazon_polarity`** because it is *exactly* the domain ("Product Review Intelligence") and gives plenty of headroom for stratified sampling.

## Subsetting strategy

3.6M is overkill for a 4-week project on free Colab. We use:

- **Train**: 50,000 stratified examples (25k pos + 25k neg)
- **Val**: 5,000 stratified
- **Test**: 5,000 stratified (held out from `test` split — never touches training)

Stratification = same class balance as the source. Done with `dataset.shuffle(seed=42).select(range(N))` per class.

## Preprocessing

1. Concatenate `title` + `content` with a separator: `f"{title}. {content}"`. Both fields are weakly informative on their own; the title acts like a summary signal.
2. Truncate to `max_length=256` tokens (DistilBERT WordPiece). Tail of long reviews is rarely informative.
3. No lowercase / no stopword removal — DistilBERT handles casing implicitly via its tokenizer.

## Label semantics

Source is binary:
- `0 = negative` (1–2 stars)
- `1 = positive` (4–5 stars)
- 3-star reviews are **excluded by the dataset authors** — there is no "neutral" class. We surface this clearly in the report so we don't oversell what the model can do.

## Reproducibility

- Random seed: `42` everywhere (datasets shuffle, torch, numpy).
- `datasets` cache is left at default (`~/.cache/huggingface/datasets`) and the chosen split sizes are pinned in `.env`.

## Risks & honest caveats

- **Domain shift**: amazon_polarity is mostly older Amazon reviews (~2015). If the user feeds 2025 reviews of a niche product (e.g. "AI-powered earbuds"), accuracy may drop. We mention this in the report and surface a confidence score.
- **Sarcasm / irony**: amazon_polarity has plenty but no ground-truth flag. BERT will mis-classify a slice of these — we keep an "edge cases" section in the report with examples.
- **English only**: training data is English; Spanish/French reviews will be misclassified. Out of scope for W1–W4.
