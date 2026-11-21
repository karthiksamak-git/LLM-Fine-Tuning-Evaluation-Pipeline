"""Trainer setup: LoRA + custom collator, mixed precision, eval, best checkpoint, resume."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from mlfte.config import Config
from mlfte.data.collator import CausalLMCollator
from mlfte.data.prepare import load_splits
from mlfte.data.tokenize import build_dataset, load_tokenizer
from mlfte.device import PrecisionSpec, resolve_device, select_precision
from mlfte.env_info import collect_env, write_json
from mlfte.logging_utils import get_logger
from mlfte.seed import seed_everything
from mlfte.tasks.base import Task
from mlfte.training.lora import (
    apply_lora,
    assert_only_lora_trainable,
    cast_trainable_to_fp32,
    parameter_accounting,
)

log = get_logger(__name__)


def adapter_dir(cfg: Config, seed: int, tag: str = "") -> Path:
    return cfg.outputs_dir / f"{tag}seed{seed}" / "adapter"


def metrics_path(cfg: Config, seed: int, tag: str = "") -> Path:
    return cfg.results_dir / "training" / f"{tag}seed{seed}.json"


class MemoryCallback(TrainerCallback):
    """Track peak process RSS (and CUDA peak) plus a wall-clock timeline."""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self.process = psutil.Process()
        self.peak_rss_mb = 0.0
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        rss = self.process.memory_info().rss / 1024**2
        self.peak_rss_mb = max(self.peak_rss_mb, rss)
        return control


def build_model(cfg: Config, precision: PrecisionSpec):
    """Load the base model in fp32 and wrap it with LoRA adapters."""
    from transformers import AutoModelForCausalLM

    device = resolve_device()
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32)
    model.to(device)
    lora_model, lora_info = apply_lora(model, cfg)
    if precision.use_autocast and precision.dtype_name == "float16":
        cast = cast_trainable_to_fp32(lora_model)
        lora_info["fp32_casts"] = cast
    if cfg.training.gradient_checkpointing:
        if hasattr(lora_model, "enable_input_require_grads"):
            lora_model.enable_input_require_grads()
        lora_model.gradient_checkpointing_enable()
    assert_only_lora_trainable(lora_model)
    lora_model.print_trainable_parameters()
    lora_info["parameter_accounting"] = parameter_accounting(lora_model)
    return lora_model, lora_info


def total_training_steps(cfg: Config, train_size: int) -> int:
    if cfg.training.max_steps and cfg.training.max_steps > 0:
        return cfg.training.max_steps
    per_epoch = math.ceil(
        train_size
        / (cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps)
    )
    return max(1, per_epoch * int(math.ceil(cfg.training.num_train_epochs)))


def build_training_arguments(
    cfg: Config, precision: PrecisionSpec, output_dir: Path, train_size: int
) -> TrainingArguments:
    steps = total_training_steps(cfg, train_size)
    warmup_steps = max(1, int(cfg.training.warmup_ratio * steps))
    eval_steps = max(1, min(cfg.training.eval_steps, steps))
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "max_steps": cfg.training.max_steps,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "max_grad_norm": cfg.training.max_grad_norm,
        "gradient_checkpointing": cfg.training.gradient_checkpointing,
        "bf16": precision.dtype_name == "bfloat16",
        "fp16": precision.dtype_name == "float16",
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": eval_steps,
        "save_total_limit": cfg.training.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": max(1, cfg.training.logging_steps),
        "logging_first_step": True,
        "report_to": [],
        "disable_tqdm": False,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": 0,
        "seed": cfg.run.seed,
        "data_seed": cfg.run.seed,
        "use_cpu": resolve_device() == "cpu",
        "optim": "adamw_torch",
        "dataloader_pin_memory": False,
    }
    return TrainingArguments(**kwargs)


def build_trainer(
    cfg: Config,
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: Path,
    precision: PrecisionSpec | None = None,
) -> tuple[Trainer, MemoryCallback]:
    precision = precision or select_precision(resolve_device(), _bf16_ok())
    args = build_training_arguments(cfg, precision, output_dir, len(train_ds))
    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    memory_cb = MemoryCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[memory_cb],
    )
    return trainer, memory_cb


def _bf16_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover
        return False


def _curves(log_history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_curve, eval_curve = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_curve.append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "learning_rate": entry.get("learning_rate"),
                    "grad_norm": entry.get("grad_norm"),
                }
            )
        if "eval_loss" in entry:
            eval_curve.append(
                {"step": entry.get("step"), "epoch": entry.get("epoch"),
                 "eval_loss": entry.get("eval_loss")}
            )
    return train_curve, eval_curve


def train(
    cfg: Config,
    task: Task,
    seed: int | None = None,
    tag: str = "",
    resume_from_checkpoint: str | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one LoRA training job and persist adapter + metrics."""
    seed = seed if seed is not None else cfg.run.train_seeds[0]
    seed_everything(seed)
    device = resolve_device()
    precision = select_precision(device, _bf16_ok())
    log.info(
        "training seed=%d device=%s precision=%s", seed, device, precision.label
    )

    splits = load_splits(cfg)
    # The ablation grid reuses the frozen split files; a smaller `data.train_size` must
    # actually shrink the training set (deterministically, from the front of the file).
    if len(splits["train"]) > cfg.data.train_size:
        log.info(
            "subsampling train split: %d -> %d examples",
            len(splits["train"]),
            cfg.data.train_size,
        )
        splits["train"] = splits["train"][: cfg.data.train_size]
    tokenizer = load_tokenizer(cfg.model.name)
    train_ds, train_stats = build_dataset(cfg, task, splits["train"], tokenizer, "train")
    val_ds, val_stats = build_dataset(cfg, task, splits["val"], tokenizer, "val")

    model, lora_info = build_model(cfg, precision)
    out_dir = adapter_dir(cfg, seed, tag)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    trainer, memory_cb = build_trainer(cfg, model, tokenizer, train_ds, val_ds, out_dir, precision)

    if max_steps is not None and max_steps > 0:
        trainer.args.max_steps = max_steps
        trainer.args.num_train_epochs = 1
        trainer.args.warmup_steps = max(1, int(cfg.training.warmup_ratio * max_steps))
        trainer.args.eval_steps = max(1, min(max_steps, cfg.training.eval_steps))
        trainer.args.save_steps = trainer.args.eval_steps

    start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    wall_clock = time.perf_counter() - start
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    avg_len = sum(len(r) for r in train_ds["input_ids"]) / len(train_ds)
    steps = trainer.state.global_step
    tokens_seen = steps * cfg.training.per_device_train_batch_size * (
        cfg.training.gradient_accumulation_steps
    ) * avg_len
    train_curve, eval_curve = _curves(trainer.state.log_history)
    best_eval = min(
        (entry["eval_loss"] for entry in eval_curve), default=None
    )
    metrics = {
        "seed": seed,
        "tag": tag,
        "profile": cfg.profile,
        "device": device,
        "precision": precision.to_dict(),
        "lora": lora_info,
        "hyperparameters": cfg.training.model_dump(),
        "effective_batch_size": (
            cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps
        ),
        "steps": steps,
        "epochs_completed": round(trainer.state.epoch or 0.0, 3),
        "wall_clock_s": round(wall_clock, 2),
        "train_loss": train_result.metrics.get("train_loss"),
        "best_eval_loss": best_eval,
        "final_train_loss": train_curve[-1]["loss"] if train_curve else None,
        "train_tokens_per_second": round(tokens_seen / wall_clock, 2) if wall_clock else None,
        "tokens_seen_estimate": int(tokens_seen),
        "peak_memory_mb": round(memory_cb.peak_rss_mb, 1),
        "cuda_peak_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
        "train_curve": train_curve,
        "eval_curve": eval_curve,
        "train_dataset": train_stats,
        "val_dataset": val_stats,
        "adapter_path": str(out_dir),
        "resumed_from": resume_from_checkpoint,
        "trainer_metrics": {
            k: v for k, v in train_result.metrics.items() if isinstance(v, (int, float))
        },
        "env": collect_env(cfg),
    }
    write_json(metrics_path(cfg, seed, tag), metrics)
    log.info(
        "done seed=%d steps=%d train_loss=%s best_eval_loss=%s %.1fs",
        seed, steps, metrics["train_loss"], metrics["best_eval_loss"], wall_clock,
    )
    del trainer, model
    return metrics


def load_training_metrics(cfg: Config, seed: int | None = None, tag: str = "") -> dict[str, Any]:
    seed = seed if seed is not None else cfg.run.train_seeds[0]
    path = metrics_path(cfg, seed, tag)
    if not path.exists():
        raise FileNotFoundError(f"no training metrics at {path}; run `mlfte train` first")
    return json.loads(path.read_text())
