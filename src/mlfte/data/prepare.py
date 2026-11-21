"""Load, clean, deduplicate and split the dataset; write a data card.

The test split is created once, from a fixed seed, and never used for training,
hyper-parameter selection or early stopping. Re-running without ``force`` reuses the
existing split files so every phase sees byte-identical data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mlfte.config import Config
from mlfte.env_info import data_fingerprint, write_json
from mlfte.logging_utils import get_logger
from mlfte.seed import seeded_generator
from mlfte.tasks.base import Example, Task
from mlfte.tasks.text2sql import normalize_whitespace

log = get_logger(__name__)

SPLIT_SEED = 1234
SPLITS = ("train", "val", "test")


def _dedupe_key(example: Example) -> str:
    payload = (
        f"{normalize_whitespace(example.question).lower()}\x00"
        f"{normalize_whitespace(example.context).lower()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_examples(examples: list[Example], cfg: Config) -> tuple[list[Example], dict[str, int]]:
    """Normalise whitespace and drop empty / over-length rows. Returns kept + drop counts."""
    drops = Counter()
    kept: list[Example] = []
    for ex in examples:
        question = normalize_whitespace(ex.question)
        context = ex.context.strip()
        answer = normalize_whitespace(ex.answer)
        if not question or not context or not answer:
            drops["empty_field"] += 1
            continue
        if len(answer) < cfg.data.min_answer_chars:
            drops["answer_too_short"] += 1
            continue
        if len(answer) > cfg.data.max_answer_chars:
            drops["answer_too_long"] += 1
            continue
        if len(context) > cfg.data.max_prompt_chars:
            drops["schema_too_long"] += 1
            continue
        kept.append(Example(ex.id, question, context, answer))
    return kept, dict(drops)


def dedupe_examples(examples: list[Example]) -> tuple[list[Example], int]:
    """Drop exact duplicates on normalised (question, schema)."""
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in examples:
        key = _dedupe_key(ex)
        if key in seen:
            continue
        seen.add(key)
        kept.append(ex)
    return kept, len(examples) - len(kept)


def split_examples(
    examples: list[Example], cfg: Config, seed: int = SPLIT_SEED
) -> dict[str, list[Example]]:
    """Deterministic split: test first, then val, then train, from one shuffle."""
    rng = seeded_generator(seed)
    order = rng.permutation(len(examples))
    shuffled = [examples[int(i)] for i in order]

    need = cfg.data.train_size + cfg.data.val_size + cfg.data.test_size
    if len(shuffled) < need:
        raise ValueError(f"not enough examples after cleaning: have {len(shuffled)}, need {need}")

    test = shuffled[: cfg.data.test_size]
    val = shuffled[cfg.data.test_size : cfg.data.test_size + cfg.data.val_size]
    train = shuffled[
        cfg.data.test_size + cfg.data.val_size : cfg.data.test_size
        + cfg.data.val_size
        + cfg.data.train_size
    ]
    return {"train": train, "val": val, "test": test}


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        f"p{p}": float(np.percentile(arr, p)) for p in (0, 25, 50, 75, 95, 99, 100)
    } | {"mean": float(arr.mean())}


def split_paths(cfg: Config) -> dict[str, Path]:
    return {name: cfg.results_dir / "data" / f"{name}.jsonl" for name in SPLITS}


def write_splits(cfg: Config, splits: dict[str, list[Example]]) -> dict[str, Path]:
    paths = split_paths(cfg)
    for name, examples in splits.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return paths


def read_split(path: str | Path) -> list[Example]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split not found: {path} (run `mlfte prepare` first)")
    with path.open(encoding="utf-8") as fh:
        return [Example.from_dict(json.loads(line)) for line in fh if line.strip()]


def load_splits(cfg: Config) -> dict[str, list[Example]]:
    return {name: read_split(path) for name, path in split_paths(cfg).items()}


def build_data_card(
    cfg: Config,
    task: Task,
    raw_count: int,
    cleaned: list[Example],
    drops: dict[str, int],
    duplicates_removed: int,
    splits: dict[str, list[Example]],
    dedupe_reports: dict[str, Any],
    token_stats: dict[str, Any] | None,
    paths: dict[str, Path],
) -> dict[str, Any]:
    complexity = Counter(task.bucket(ex) for ex in splits["train"])
    per_split = {}
    for name, examples in splits.items():
        per_split[name] = {
            "count": len(examples),
            "question_chars": _percentiles([len(e.question) for e in examples]),
            "schema_chars": _percentiles([len(e.context) for e in examples]),
            "answer_chars": _percentiles([len(e.answer) for e in examples]),
            "complexity": dict(Counter(task.bucket(e) for e in examples)),
        }
    return {
        "schema_version": 1,
        "profile": cfg.profile,
        "task": task.name,
        "dataset": {
            "name": cfg.data.dataset,
            "license": "CC-BY-4.0",
            "attribution": (
                "b-mc2/sql-create-context on Hugging Face Datasets. Prompt/answer pairs "
                "derived from the Spider and WikiSQL text-to-SQL corpora."
            ),
            "raw_rows": raw_count,
        },
        "cleaning": {
            "kept": len(cleaned),
            "dropped": drops,
            "exact_duplicates_removed": duplicates_removed,
            "min_answer_chars": cfg.data.min_answer_chars,
            "max_answer_chars": cfg.data.max_answer_chars,
            "max_prompt_chars": cfg.data.max_prompt_chars,
        },
        "split": {
            "seed": SPLIT_SEED,
            "sizes_requested": {
                "train": cfg.data.train_size,
                "val": cfg.data.val_size,
                "test": cfg.data.test_size,
            },
            "policy": (
                "Single shuffle from a fixed seed; test is carved out first, then val, then "
                "train. The test split is never used for training, model selection or early "
                "stopping."
            ),
        },
        "splits": per_split,
        "total_complexity": dict(complexity),
        "decontamination": dedupe_reports,
        "prompt_template": {
            "system": task.system_prompt(),
            "user": task.user_prompt(splits["train"][0]),
            "assistant": task.target(splits["train"][0]),
            "rendering": "tokenizer.apply_chat_template(messages, add_generation_prompt=True)",
        },
        "token_stats": token_stats,
        "data_fingerprint": data_fingerprint(list(paths.values())),
    }


def prepare(cfg: Config, task: Task, force: bool = False) -> dict[str, Any]:
    """Build splits + data card. Reuses existing splits unless ``force``."""
    from mlfte.data.decontam import decontaminate
    from mlfte.data.tokenize import token_stats_for_splits

    paths = split_paths(cfg)
    card_path = cfg.results_dir / "data" / "data_card.json"
    if not force and card_path.exists() and all(p.exists() for p in paths.values()):
        log.info("data card and splits already exist, reusing (%s)", card_path)
        return json.loads(card_path.read_text())

    raw = task.load_examples(cfg)
    log.info("loaded %d raw rows from %s", len(raw), cfg.data.dataset)
    cleaned, drops = clean_examples(raw, cfg)
    deduped, duplicates_removed = dedupe_examples(cleaned)
    log.info(
        "cleaned=%d dropped=%s duplicates_removed=%d", len(deduped), drops, duplicates_removed
    )
    splits = split_examples(deduped, cfg)
    write_splits(cfg, splits)
    log.info("splits written: %s", {k: len(v) for k, v in splits.items()})

    reports = decontaminate(splits)
    if not reports["disjoint"]:
        raise RuntimeError(f"splits are not disjoint: {reports}")

    token_stats = token_stats_for_splits(cfg, task, splits)
    card = build_data_card(
        cfg,
        task,
        len(raw),
        deduped,
        drops,
        duplicates_removed,
        splits,
        reports,
        token_stats,
        paths,
    )
    write_json(card_path, card)
    log.info("wrote data card -> %s", card_path)
    return card
