"""Evaluate the saved model on the test split: metrics + confusion matrix.

Run:
    python -m src.model.evaluate
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import cfg
from src.model.dataset import load_splits
from src.utils.logging_config import configure_logging, get_logger


def main() -> None:
    configure_logging(run_id="eval")
    log = get_logger("model.eval")

    if not (cfg.model_dir / "config.json").exists():
        raise FileNotFoundError(f"No model found at {cfg.model_dir}. Run `python -m src.model.train` first.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_dir)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    log.info("eval.device", extra={"device": device})

    _, _, test = load_splits(seed=42)

    all_preds, all_labels = [], []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(test), batch_size):
            batch = test[i : i + batch_size]
            inputs = tokenizer(
                batch["text"],
                truncation=True,
                padding=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"])

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=cfg.label_names(), digits=3)
    cm = confusion_matrix(all_labels, all_preds)

    log.info("eval.metrics", extra={"accuracy": acc, "f1_weighted": f1})
    print(report)

    metrics = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_test": len(all_labels),
    }
    (cfg.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (cfg.output_dir / "eval_classification_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=cfg.label_names(), yticklabels=cfg.label_names(),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — acc={acc:.3f}, F1={f1:.3f}")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "eval_confusion_matrix.png", dpi=150)
    log.info("eval.saved", extra={"output_dir": str(cfg.output_dir)})


if __name__ == "__main__":
    main()
