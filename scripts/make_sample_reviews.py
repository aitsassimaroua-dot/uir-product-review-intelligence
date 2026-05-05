"""Create a small CSV of sample product reviews for the demo.

Run:
    python scripts/make_sample_reviews.py
Writes:
    data/processed/sample_reviews.csv  (10 reviews, mixed sentiment)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "sample_reviews.csv"

REVIEWS = [
    (1, "Battery dies in two hours, totally useless for travel.", 1),
    (2, "Sound quality is amazing, way better than my old AirPods.", 5),
    (3, "Stopped charging after one week. Waste of money.", 1),
    (4, "Comfortable fit, great noise cancellation, love them.", 5),
    (5, "App is buggy and keeps disconnecting from my phone.", 2),
    (6, "Excellent value for the price, would buy again.", 5),
    (7, "Bass is decent but the case is flimsy and cracked.", 2),
    (8, "Perfect for my morning runs, sweat-proof as advertised.", 5),
    (9, "Customer support never replied to my refund request.", 1),
    (10, "Surprised by the build quality at this price point.", 4),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "rating"])
        w.writerows(REVIEWS)
    print(f"✓ Wrote {len(REVIEWS)} sample reviews to {OUT}")


if __name__ == "__main__":
    main()
