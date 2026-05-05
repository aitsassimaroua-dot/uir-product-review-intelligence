"""Fine-tune DistilBERT on amazon_polarity.

Run:
    python -m src.model.train
    python -m src.model.train --epochs 1 --subset 10000   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import cfg
from src.model.dataset import load_splits, tokenize_dataset
from src.utils.logging_config import configure_logging, get_logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1,
        "precision_weighted": p,
        "recall_weighted": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--subset", type=int, default=cfg.train_subset, help="Override TRAIN_SUBSET.")
    args = parser.parse_args()

    configure_logging(run_id=f"train_{Path(cfg.model_dir).name}")
    log = get_logger("model.train")
    set_seed(42)

    # Honor CLI subset override by patching cfg at runtime.
    if args.subset != cfg.train_subset:
        log.info("train.subset_override", extra={"requested": args.subset})
        # cfg is frozen — just rebuild the splits with the override applied locally.
        import src.config as _c
        object.__setattr__(_c.cfg, "train_subset", args.subset)

    log.info("train.start", extra={
        "base_model": cfg.base_model, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
    })

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    train, val, test = load_splits(seed=42)
    train_t = tokenize_dataset(train, tokenizer)
    val_t = tokenize_dataset(val, tokenizer)
    test_t = tokenize_dataset(test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        id2label={i: name for i, name in enumerate(cfg.label_names())},
        label2id={name: i for i, name in enumerate(cfg.label_names())},
    )

    out_dir = cfg.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "_hf_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    # transformers 5.x: `tokenizer=` was renamed to `processing_class=`.
    # Use whichever the installed version supports.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_t,
        eval_dataset=val_t,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    import inspect
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=test_t, metric_key_prefix="test")
    log.info("train.test_metrics", extra={"metrics": {k: float(v) for k, v in test_metrics.items()}})

    # Save final model + tokenizer at cfg.model_dir (not under _hf_trainer).
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    log.info("train.saved", extra={"path": str(out_dir)})


if __name__ == "__main__":
    main()
