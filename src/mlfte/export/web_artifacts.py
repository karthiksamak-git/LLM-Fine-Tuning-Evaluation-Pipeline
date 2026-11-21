"""Turn ``results/<profile>/`` into static JSON for the Next.js dashboard.

Every file carries ``schema_version``, ``profile``, ``hardware``, ``git_commit`` and
``data_fingerprint`` so the UI can always say where a number came from.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlfte.config import Config
from mlfte.env_info import environment_badge, git_commit, package_versions, read_json, write_json
from mlfte.logging_utils import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1
RUN_LABELS = ("base_zero_shot", "base_few_shot", "finetuned_adapter", "finetuned_merged")
ARTIFACT_FILES = (
    "summary.json",
    "comparison.json",
    "training.json",
    "ablations.json",
    "efficiency.json",
    "forgetting.json",
    "error_analysis.json",
    "examples.json",
    "data_card.json",
    "model_card.json",
    "env.json",
)
COMMON_FIELDS = ("schema_version", "profile", "hardware", "git_commit")


def validate_payload(name: str, payload: Any) -> None:
    """Minimal structural contract shared by export, ingest and the test suite."""
    if not isinstance(payload, dict):
        raise ValueError(f"{name}: expected a JSON object, got {type(payload).__name__}")
    missing = [field for field in COMMON_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{name}: missing required fields {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{name}: schema_version {payload['schema_version']} != {SCHEMA_VERSION}")
    if payload["profile"] not in ("cpu", "gpu", "smoke"):
        raise ValueError(f"{name}: unknown profile {payload['profile']!r}")
    if not isinstance(payload["hardware"], dict):
        raise ValueError(f"{name}: hardware must be an object")
    if name == "examples.json" and not isinstance(payload.get("examples"), list):
        raise ValueError("examples.json: examples must be a list")
    if name == "summary.json" and not isinstance(payload.get("headline"), dict):
        raise ValueError("summary.json: headline must be an object")
    if name == "comparison.json" and not isinstance(payload.get("runs", {}), dict):
        raise ValueError("comparison.json: runs must be an object")

CPU_NOTICE = (
    "Only the CPU profile has been run. Model: SmolLM2-135M-Instruct (134.5M params), fp32, "
    "no mixed precision, on a 2 vCPU / 3 GB RAM machine. These numbers are NOT GPU results "
    "and do not describe Qwen2.5-0.5B or any bf16/fp16 run."
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return default


def common_meta(cfg: Config) -> dict[str, Any]:
    data_dir = cfg.results_dir / "data"
    card = _load(data_dir / "data_card.json", {}) or {}
    metrics = _load(cfg.results_dir / "training" / f"{cfg.run.tag}_seed{cfg.run.train_seeds[0]}.json")
    if metrics is None:
        candidate = sorted((cfg.results_dir / "training").glob("*.json"))
        metrics = _load(candidate[0]) if candidate else {}
    env = (metrics or {}).get("env", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": cfg.profile,
        "task": "text_to_sql",
        "dataset": cfg.data.dataset,
        "model": cfg.model.name,
        "hardware": environment_badge(env if env else {"profile": cfg.profile}),
        "raw_hardware": (env or {}).get("hardware", {}),
        "git_commit": git_commit(),
        "data_fingerprint": card.get("data_fingerprint"),
        "generated_at": _now(),
        "package_versions": package_versions(),
    }


def _run_metrics(cfg: Config) -> dict[str, Any]:
    return _load(cfg.results_dir / "eval" / "runs.json", {}) or {}


def build_comparison(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    runs = _load(cfg.results_dir / "comparison.json", {}) or {}
    return {**meta, **runs}


def build_training(cfg: Config, meta: dict[str, Any], seed: int) -> dict[str, Any]:
    path = cfg.results_dir / "training" / f"seed{seed}.json"
    metrics = _load(path, {}) or {}
    seed_summary = _load(cfg.results_dir / "seeds.json", {}) or {}
    return {
        **meta,
        "seed": seed,
        "curves": {
            "train": metrics.get("train_curve", []),
            "eval": metrics.get("eval_curve", []),
        },
        "hyperparameters": metrics.get("hyperparameters", {}),
        "lora": metrics.get("lora", {}),
        "throughput": {
            "train_tokens_per_second": metrics.get("train_tokens_per_second"),
            "train_samples_per_second": (metrics.get("trainer_metrics") or {}).get(
                "train_samples_per_second"
            ),
            "steps": metrics.get("steps"),
            "epochs_completed": metrics.get("epochs_completed"),
            "wall_clock_s": metrics.get("wall_clock_s"),
            "effective_batch_size": metrics.get("effective_batch_size"),
        },
        "memory": {
            "peak_rss_mb": metrics.get("peak_memory_mb"),
            "cuda_peak_allocated_mb": metrics.get("cuda_peak_memory_mb"),
        },
        "precision": metrics.get("precision", {}),
        "device": metrics.get("device"),
        "train_dataset": metrics.get("train_dataset", {}),
        "val_dataset": metrics.get("val_dataset", {}),
        "seeds": seed_summary,
    }


def build_ablations(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load(cfg.results_dir / "ablations.json", {}) or {}
    payload = {**meta, **data}
    payload.setdefault("points", [])
    if not payload["points"]:
        payload.setdefault(
            "note",
            "No ablation points were recorded for this profile yet. Run `mlfte ablate` "
            "and re-export to populate this page.",
        )
    return payload


def build_efficiency(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load(cfg.results_dir / "efficiency.json", {}) or {}
    return {**meta, **data}


def build_forgetting(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load(cfg.results_dir / "forgetting.json", {}) or {}
    return {**meta, **data}


def build_error_analysis(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    data = _load(cfg.results_dir / "error_analysis.json", {}) or {}
    return {**meta, **data}


def build_examples(cfg: Config, meta: dict[str, Any], limit: int = 200) -> dict[str, Any]:
    analysis = _load(cfg.results_dir / "error_analysis.json", {}) or {}
    examples = analysis.get("examples", [])[:limit]
    return {**meta, "count": len(examples), "examples": examples}


def build_data_card(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    card = _load(cfg.results_dir / "data" / "data_card.json", {}) or {}
    return {**meta, **card}


def build_model_card(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    seed = cfg.run.train_seeds[0]
    metrics = _load(cfg.results_dir / "training" / f"seed{seed}.json", {}) or {}
    parity = _load(cfg.results_dir / "merge" / "parity.json", {}) or {}
    hub = _load(cfg.results_dir / "hub.json", {}) or {}
    lora = metrics.get("lora", {})
    return {
        **meta,
        "base_model": cfg.model.name,
        "lora": {
            "r": lora.get("lora_r", cfg.lora.r),
            "alpha": lora.get("lora_alpha", cfg.lora.alpha),
            "dropout": lora.get("lora_dropout", cfg.lora.dropout),
            "target_modules": lora.get("target_modules", []),
            "trainable_parameters": lora.get("trainable_parameters"),
            "total_parameters": lora.get("total_parameters"),
            "trainable_percent": lora.get("trainable_percent"),
            "task_type": lora.get("task_type"),
        },
        "hyperparameters": metrics.get("hyperparameters", {}),
        "seeds": cfg.run.train_seeds,
        "precision": metrics.get("precision", {}),
        "device": metrics.get("device"),
        "merge_parity": parity,
        "hub_url": hub.get("url"),
        "hub_published": bool(hub.get("url")),
        "limitations": _limitations(cfg),
    }


def _limitations(cfg: Config) -> list[str]:
    out = [
        "Single task (text-to-SQL on b-mc2/sql-create-context); no claim of generality.",
        "Execution match runs on synthetic seeded SQLite tables, not the original databases.",
        "Small open-weight base model; absolute numbers are low by design.",
    ]
    if cfg.profile == "cpu":
        out.insert(0, CPU_NOTICE)
    else:
        out.insert(0, "Numbers come from the profile/hardware recorded in `hardware`.")
    if cfg.data.train_size <= 2000:
        out.append(
            f"Trained on a {cfg.data.train_size}-example subset, not the full 78k-row dataset."
        )
    return out


def build_summary(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    comparison = _load(cfg.results_dir / "comparison.json", {}) or {}
    runs = comparison.get("runs", {})
    finetuned = "finetuned_adapter" if "finetuned_adapter" in runs else "finetuned_merged"
    headline_metric = "execution_match"
    headline: dict[str, Any] = {"primary_metric": headline_metric, "test_n": None}
    for label, payload in runs.items():
        if payload.get("metrics"):
            headline.setdefault("execution_match", {})[label] = payload["metrics"][
                headline_metric
            ]["value"]
            headline.setdefault("exact_match", {})[label] = payload["metrics"]["exact_match"][
                "value"
            ]
            headline["test_n"] = payload["metrics"][headline_metric]["n"]
    if headline.get("execution_match") and finetuned in headline["execution_match"]:
        head = headline["execution_match"]
        baselines = {k: v for k, v in head.items() if k.startswith("base_")}
        if baselines:
            best = max(baselines, key=baselines.get)
            cmp = comparison.get("comparisons", {}).get(f"{finetuned}__vs__{best}", {})
            metric_cmp = cmp.get(headline_metric, {})
            headline["best_baseline"] = best
            headline["gain_over_best_baseline"] = {
                "absolute": metric_cmp.get("absolute_difference"),
                "relative": metric_cmp.get("relative_improvement"),
                "p_value": (metric_cmp.get("mcnemar") or {}).get("p_value"),
                "significant_at_0_05": (
                    (metric_cmp.get("mcnemar") or {}).get("p_value") is not None
                    and (metric_cmp.get("mcnemar") or {}).get("p_value") < 0.05
                ),
            }
    training = _load(cfg.results_dir / "training" / f"seed{cfg.run.train_seeds[0]}.json", {}) or {}
    return {
        **meta,
        "headline": headline,
        "profile_notice": CPU_NOTICE if cfg.profile == "cpu" else None,
        "training": {
            "steps": training.get("steps"),
            "epochs_completed": training.get("epochs_completed"),
            "wall_clock_s": training.get("wall_clock_s"),
            "train_tokens_per_second": training.get("train_tokens_per_second"),
            "precision": (training.get("precision") or {}).get("label"),
            "trainable_percent": (training.get("lora") or {}).get("trainable_percent"),
        },
        "splits": {
            "train": cfg.data.train_size,
            "val": cfg.data.val_size,
            "test": cfg.data.test_size,
        },
        "pipeline_steps": [
            "prepare: clean, dedupe, fixed splits, decontamination",
            "baseline: base model zero-shot and 3-shot",
            "train: LoRA with completion-only loss masking",
            "eval: exact match, validity, execution match with bootstrap CIs",
            "ablate: rank / lr / target modules / data size / loss masking",
            "analyze: failure taxonomy and curated examples",
            "export: static JSON rendered by this dashboard",
        ],
        "limitations": _limitations(cfg),
    }


def build_env(cfg: Config, meta: dict[str, Any]) -> dict[str, Any]:
    seed = cfg.run.train_seeds[0]
    metrics = _load(cfg.results_dir / "training" / f"seed{seed}.json", {}) or {}
    return {**meta, "env": metrics.get("env", {}), "config": cfg.to_dict()}


def build_all(cfg: Config, out_dir: Path | None = None) -> dict[str, Path]:
    """Write every web artifact, plus ``results/export/`` for ingest/zipping."""
    out_dir = out_dir or Path(__file__).resolve().parents[3] / "web" / "public" / "data"
    export_dir = cfg.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.run.train_seeds[0]
    meta = common_meta(cfg)

    artifacts = {
        "summary.json": build_summary(cfg, meta),
        "comparison.json": build_comparison(cfg, meta),
        "training.json": build_training(cfg, meta, seed),
        "ablations.json": build_ablations(cfg, meta),
        "efficiency.json": build_efficiency(cfg, meta),
        "forgetting.json": build_forgetting(cfg, meta),
        "error_analysis.json": build_error_analysis(cfg, meta),
        "examples.json": build_examples(cfg, meta),
        "data_card.json": build_data_card(cfg, meta),
        "model_card.json": build_model_card(cfg, meta),
        "env.json": build_env(cfg, meta),
    }
    written: dict[str, Path] = {}
    total = 0
    for name, payload in artifacts.items():
        validate_payload(name, payload)
        path = write_json(out_dir / name, payload)
        write_json(export_dir / name, payload)
        total += path.stat().st_size
        written[name] = path
        log.info("exported %s (%.1f KB)", name, path.stat().st_size / 1024)
    log.info("total web data: %.2f MB", total / 1024**2)
    if total > 5 * 1024**2:
        raise RuntimeError(f"web data exceeds 5 MB ({total / 1024**2:.2f} MB)")
    # Keep the raw data card JSON available for the ingest manifest too.
    card = cfg.results_dir / "data" / "data_card.json"
    if card.exists():
        shutil.copy2(card, export_dir / "data_card_raw.json")
    return written


def load_artifact(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / "web" / "public" / "data" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())
