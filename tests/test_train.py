"""Trainer integration on a tiny random model: overfit, checkpoint, resume, merge parity."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from mlfte.config import load_config
from mlfte.data.collator import CausalLMCollator
from mlfte.data.tokenize import encode_example
from mlfte.device import select_precision
from mlfte.tasks import get_task
from mlfte.tasks.base import Example
from mlfte.training.lora import apply_lora, assert_only_lora_trainable
from mlfte.training.train import build_trainer

EXAMPLES = [
    Example("a", "How many rows?", "CREATE TABLE t (a INTEGER);", "SELECT COUNT(*) FROM t"),
    Example("b", "Show names", "CREATE TABLE t (name TEXT);", "SELECT name FROM t"),
    Example("c", "Max age", "CREATE TABLE t (age INTEGER);", "SELECT MAX(age) FROM t"),
    Example("d", "Only small", "CREATE TABLE t (a INTEGER);", "SELECT a FROM t WHERE a < 5"),
]


def _dataset(tokenizer, max_seq_len: int = 64):
    from datasets import Dataset

    task = get_task()
    records = [encode_example(tokenizer, task, ex, max_seq_len)[0] for ex in EXAMPLES]
    return Dataset.from_list(
        [{"input_ids": r["input_ids"], "labels": r["labels"], "attention_mask": r["attention_mask"]}
         for r in records]
    )


def _trainer(tmp_path, tokenizer, model, max_steps: int, lr: float = 5e-3, seed: int = 0):
    cfg = load_config(profile="smoke")
    cfg.training.per_device_train_batch_size = 1
    cfg.training.gradient_accumulation_steps = 1
    cfg.training.max_steps = max_steps
    cfg.training.learning_rate = lr
    cfg.training.eval_steps = max(1, max_steps)
    cfg.training.logging_steps = 1
    cfg.training.save_total_limit = 1
    cfg.training.lr_scheduler_type = "constant"
    cfg.run.seed = seed
    dataset = _dataset(tokenizer)
    trainer, _ = build_trainer(
        cfg, model, tokenizer, dataset, dataset, Path(tmp_path) / "out",
        select_precision("cpu", False),
    )
    return trainer


def test_loss_decreases_on_tiny_overfit_batch(tmp_path, tokenizer, tiny_model):
    """The loop must actually optimise: smoothed loss falls on a repeated tiny batch."""
    torch.manual_seed(1234)
    model, _ = apply_lora(tiny_model, load_config(profile="smoke"))
    assert_only_lora_trainable(model)
    trainer = _trainer(tmp_path, tokenizer, model, max_steps=30)
    trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    assert len(losses) >= 20, "trainer logged too few losses"
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first * 0.96, f"smoothed loss did not decrease: {first:.3f} -> {last:.3f}"
    assert torch.isfinite(torch.tensor(losses[-1]))


def test_checkpoint_roundtrip_and_resume(tmp_path, tokenizer, tiny_model):
    cfg = load_config(profile="smoke")
    torch.manual_seed(0)
    model, _ = apply_lora(copy.deepcopy(tiny_model), cfg)
    trainer = _trainer(tmp_path / "first", tokenizer, model, max_steps=6)
    trainer.train()
    checkpoint = Path(tmp_path) / "first" / "manual_ckpt"
    trainer.save_model(str(checkpoint))
    trainer.save_state()
    state_file = Path(trainer.args.output_dir) / "trainer_state.json"
    (checkpoint / "trainer_state.json").write_text(state_file.read_text())
    assert (checkpoint / "trainer_state.json").exists()

    torch.manual_seed(0)
    fresh = apply_lora(copy.deepcopy(tiny_model), cfg)[0]
    resumed = _trainer(tmp_path / "second", tokenizer, fresh, max_steps=10)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 10
    losses = [e["loss"] for e in resumed.state.log_history if "loss" in e]
    assert losses and all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_resumed_run_matches_continuous_run(tmp_path, tokenizer, tiny_model):
    cfg = load_config(profile="smoke")

    torch.manual_seed(0)
    continuous = apply_lora(copy.deepcopy(tiny_model), cfg)[0]
    trainer_a = _trainer(tmp_path / "continuous", tokenizer, continuous, max_steps=8)
    trainer_a.train()
    loss_a = [e["loss"] for e in trainer_a.state.log_history if "loss" in e][-1]

    torch.manual_seed(0)
    split = apply_lora(copy.deepcopy(tiny_model), cfg)[0]
    trainer_b = _trainer(tmp_path / "split", tokenizer, split, max_steps=4)
    trainer_b.train()
    ckpt = Path(tmp_path) / "split" / "ckpt"
    trainer_b.save_model(str(ckpt))
    trainer_b.save_state()
    state_file = Path(trainer_b.args.output_dir) / "trainer_state.json"
    (ckpt / "trainer_state.json").write_text(state_file.read_text())

    torch.manual_seed(0)
    resumed_model = apply_lora(copy.deepcopy(tiny_model), cfg)[0]
    trainer_c = _trainer(tmp_path / "resumed", tokenizer, resumed_model, max_steps=8)
    trainer_c.train(resume_from_checkpoint=str(ckpt))
    loss_c = [e["loss"] for e in trainer_c.state.log_history if "loss" in e][-1]
    assert trainer_c.state.global_step == 8
    assert abs(loss_a - loss_c) < 0.5 * max(abs(loss_a), 1e-3)


def test_merge_parity_on_tiny_model(tmp_path, tokenizer, tiny_model):
    cfg = load_config(profile="smoke")
    torch.manual_seed(0)
    base = copy.deepcopy(tiny_model)
    model, _ = apply_lora(base, cfg)
    for param in model.parameters():
        if param.requires_grad:
            param.data.add_(torch.randn_like(param) * 0.01)
    model.eval()
    ids = torch.randint(100, 1000, (2, 16))
    with torch.inference_mode():
        before = model(input_ids=ids).logits
    merged = model.merge_and_unload()
    with torch.inference_mode():
        after = merged(input_ids=ids).logits
    assert torch.allclose(before, after, atol=5e-3)


def test_collator_batches_tokenized_examples(tokenizer):
    task = get_task()
    records = [encode_example(tokenizer, task, ex, 64)[0] for ex in EXAMPLES]
    batch = CausalLMCollator(tokenizer.pad_token_id)(records)
    assert batch["input_ids"].shape[0] == len(EXAMPLES)
    assert batch["labels"].shape == batch["input_ids"].shape
