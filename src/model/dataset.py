"""Dataset loading + tokenization for amazon_polarity.

Returns Hugging Face `Dataset` objects ready for `Trainer`. Stratified
subsetting is applied so the class distribution matches the source split.
"""
from __future__ import annotations

from typing import Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import cfg
from src.utils.logging_config import get_logger

log = get_logger("model.dataset")


def _stratified_subset(ds: Dataset, label_col: str, total: int, seed: int = 42) -> Dataset:
    """Pick `total` rows with the same class balance as `ds`."""
    classes = sorted(set(ds[label_col]))
    per_class = total // len(classes)
    parts = []
    for c in classes:
        sub = ds.filter(lambda r, c=c: r[label_col] == c).shuffle(seed=seed).select(range(per_class))
        parts.append(sub)
    from datasets import concatenate_datasets
    return concatenate_datasets(parts).shuffle(seed=seed)


def load_splits(seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """Returns (train, val, test) HF Datasets."""
    log.info("dataset.loading", extra={"dataset": cfg.hf_dataset})
    raw = load_dataset(cfg.hf_dataset)

    # amazon_polarity has columns: title, content, label
    # Subset FIRST (cheap), then map (expensive) — the reverse order would
    # run _concat on 3.6M rows for nothing.
    train_sub = _stratified_subset(raw["train"], "label", cfg.train_subset, seed)
    test_sub = _stratified_subset(raw["test"], "label", cfg.eval_subset, seed)

    def _concat(row):
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        return {"text": f"{title}. {content}".strip(". ")}

    train_sub = train_sub.map(_concat)
    test_sub = test_sub.map(_concat)

    # Carve val out of the train subset (10%) — keeps test fully held out.
    val_size = max(1, cfg.train_subset // 10)
    train_split = train_sub.train_test_split(test_size=val_size, seed=seed, stratify_by_column="label")
    train, val = train_split["train"], train_split["test"]
    test = test_sub

    log.info(
        "dataset.ready",
        extra={"train": len(train), "val": len(val), "test": len(test), "labels": cfg.num_labels},
    )
    return train, val, test


def tokenize_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=cfg.max_length,
        )
    return ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names if c not in {"label"}])
