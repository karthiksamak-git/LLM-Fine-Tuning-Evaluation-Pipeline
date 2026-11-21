"""Data pipeline: cleaning, deterministic splits, disjointness, masking, collation."""

from __future__ import annotations

import json

import pytest
import torch

from mlfte.config import load_config
from mlfte.data.collator import IGNORE_INDEX, CausalLMCollator
from mlfte.data.decontam import decontaminate, jaccard, shingles
from mlfte.data.prepare import (
    clean_examples,
    dedupe_examples,
    read_split,
    split_examples,
)
from mlfte.data.tokenize import answer_token_ids, encode_example, prompt_token_ids
from mlfte.tasks import get_task
from mlfte.tasks.base import Example


def _examples(n: int = 60) -> list[Example]:
    return [
        Example(
            id=f"ex{i}",
            question=f"question number {i}",
            context=f"CREATE TABLE t{i} (a INTEGER);",
            answer=f"SELECT a FROM t{i} WHERE a > {i}",
        )
        for i in range(n)
    ]


def test_clean_drops_overlength_and_empty():
    cfg = load_config(profile="smoke")
    cfg.data.min_answer_chars = 5
    cfg.data.max_answer_chars = 20
    examples = [
        Example("a", "q", "CREATE TABLE t (a INT);", "SELECT a FROM t"),  # 15 chars, ok
        Example("b", "q", "CREATE TABLE t (a INT);", "SELECT a FROM t1 t2 t3 t4 t5"),  # long
        Example("c", "", "CREATE TABLE t (a INT);", "SELECT a FROM t"),
    ]
    kept, drops = clean_examples(examples, cfg)
    assert [e.id for e in kept] == ["a"]
    assert drops == {"answer_too_long": 1, "empty_field": 1}


def test_dedupe_removes_exact_duplicates():
    examples = [
        Example("a", "How many?", "CREATE TABLE t (a INT);", "SELECT 1"),
        Example("b", "How   many?", "CREATE TABLE t (a INT); ", "SELECT 2"),
        Example("c", "Other?", "CREATE TABLE t (a INT);", "SELECT 3"),
    ]
    kept, removed = dedupe_examples(examples)
    assert removed == 1
    assert len(kept) == 2


def test_split_is_deterministic_and_sized():
    cfg = load_config(profile="smoke")
    cfg.data.train_size, cfg.data.val_size, cfg.data.test_size = 30, 10, 12
    examples = _examples(60)
    first = split_examples(examples, cfg)
    second = split_examples(examples, cfg)
    assert [e.id for e in first["train"]] == [e.id for e in second["train"]]
    assert len(first["train"]) == 30
    assert len(first["val"]) == 10
    assert len(first["test"]) == 12


def test_split_raises_when_too_small():
    cfg = load_config(profile="smoke")
    cfg.data.train_size, cfg.data.val_size, cfg.data.test_size = 50, 50, 50
    with pytest.raises(ValueError):
        split_examples(_examples(60), cfg)


def test_splits_are_disjoint():
    cfg = load_config(profile="smoke")
    cfg.data.train_size, cfg.data.val_size, cfg.data.test_size = 30, 10, 12
    splits = split_examples(_examples(60), cfg)
    report = decontaminate(splits)
    assert report["disjoint"] is True
    for pair in report["pairs"].values():
        assert pair["exact_overlap"] == 0
        assert pair["near_duplicate_pairs"] == 0


def test_decontam_detects_exact_and_near_duplicates():
    base = Example("a", "How many heads older than 56", "CREATE TABLE head (age INT);", "SELECT 1")
    leak = Example("b", "How many heads older than 56", "CREATE TABLE head (age INT);", "SELECT 1")
    near = Example(
        "c", "how many heads older than 56?", "create table head (age int);", "SELECT 1"
    )
    report = decontaminate({"train": [base], "test": [leak, near]})
    assert report["disjoint"] is False
    assert report["pairs"]["train__test"]["exact_overlap"] >= 1
    assert jaccard(shingles(base), shingles(near)) > 0.9


def test_write_and_read_splits_roundtrip(tmp_path):
    splits = {"train": _examples(3), "val": _examples(2), "test": _examples(1)}
    from pathlib import Path

    for name, examples in splits.items():
        path = Path(tmp_path) / f"{name}.jsonl"
        path.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in examples))
        assert [e.id for e in read_split(path)] == [e.id for e in examples]


def test_tokenization_masks_prompt_and_keeps_answer(tokenizer):
    task = get_task()
    example = Example(
        "x", "How many rows?", "CREATE TABLE t (a INTEGER);", "SELECT COUNT(*) FROM t"
    )
    record, reason = encode_example(tokenizer, task, example, max_seq_len=256)
    assert reason is None
    prompt_len = len(prompt_token_ids(tokenizer, task, example))
    answer_len = len(answer_token_ids(tokenizer, task, example))
    assert record["prompt_len"] == prompt_len
    assert record["answer_len"] == answer_len
    labels = record["labels"]
    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == record["input_ids"][prompt_len:]
    assert labels[-1] == tokenizer.eos_token_id
    assert all(lab != IGNORE_INDEX for lab in labels[prompt_len:])


def test_full_sequence_masking_keeps_all_labels(tokenizer):
    task = get_task()
    example = Example("x", "Q?", "CREATE TABLE t (a INTEGER);", "SELECT 1 FROM t")
    record, _ = encode_example(tokenizer, task, example, 256, loss_masking="full")
    assert record["labels"] == record["input_ids"]


def test_truncation_keeps_the_answer(tokenizer):
    task = get_task()
    long_schema = "CREATE TABLE t (" + ", ".join(f"c{i} INTEGER" for i in range(400)) + ");"
    example = Example("x", "How many?", long_schema, "SELECT COUNT(*) FROM t")
    record, reason = encode_example(tokenizer, task, example, max_seq_len=128)
    assert reason is None
    assert record["truncated"] is True
    assert len(record["input_ids"]) <= 128
    answer_len = record["answer_len"]
    assert record["labels"][-answer_len:] == record["input_ids"][-answer_len:]


def test_collator_pads_dynamically_and_masks_labels():
    collator = CausalLMCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 1]},
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"][0].tolist() == [-100, 2, 3]
    assert batch["labels"][1].tolist() == [-100, 5, IGNORE_INDEX]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["input_ids"].dtype == torch.long


def test_collator_rejects_empty_batch():
    with pytest.raises(ValueError):
        CausalLMCollator(pad_token_id=0)([])


def test_complexity_buckets_are_ordered():
    task = get_task()
    simple = Example("s", "q", "CREATE TABLE t (a INT);", "SELECT a FROM t")
    medium = Example("m", "q", "CREATE TABLE t (a INT);", "SELECT COUNT(a) FROM t")
    complex_q = Example(
        "c",
        "q",
        "CREATE TABLE t (a INT); CREATE TABLE u (a INT);",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE t.a IN (SELECT a FROM u) GROUP BY t.a",
    )
    assert task.bucket(simple) == "simple"
    assert task.bucket(medium) == "medium"
    assert task.bucket(complex_q) == "complex"
