"""Final figures (200 dpi, colorblind-safe) generated from the exported JSON only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlfte.config import Config, repo_root  # noqa: E402
from mlfte.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
LABEL_TEXT = {
    "base_zero_shot": "Base (zero-shot)",
    "base_few_shot": "Base (3-shot)",
    "finetuned_adapter": "Fine-tuned (LoRA)",
    "finetuned_merged": "Fine-tuned (merged)",
}
RUN_ORDER = ["base_zero_shot", "base_few_shot", "finetuned_adapter", "finetuned_merged"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.autolayout": True,
        }
    )


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("figure %s (%.0f KB)", path.name, path.stat().st_size / 1024)
    return path


def plot_metrics_comparison(comparison: dict[str, Any], out: Path) -> Path | None:
    runs = comparison.get("runs", {})
    labels = [label for label in RUN_ORDER if label in runs]
    if not labels:
        return None
    metric_keys = [("exact_match", "Exact match"), ("is_valid", "Valid SQL"),
                   ("execution_match", "Execution match")]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    width = 0.8 / len(metric_keys)
    for i, (key, title) in enumerate(metric_keys):
        values = [runs[run_label]["metrics"][key]["value"] for run_label in labels]
        lows = [runs[run_label]["metrics"][key]["value"] - runs[run_label]["metrics"][key]["ci_low"] for run_label in labels]
        highs = [runs[run_label]["metrics"][key]["ci_high"] - runs[run_label]["metrics"][key]["value"] for run_label in labels]
        positions = [j + i * width - 0.4 + width / 2 for j in range(len(labels))]
        ax.bar(positions, values, width=width, label=title, color=OKABE_ITO[i],
               yerr=[lows, highs], capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([LABEL_TEXT.get(run_label, run_label) for run_label in labels], rotation=12)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Text-to-SQL on the held-out test split ({comparison.get('profile')} profile)")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    return _save(fig, out)


def plot_training_curves(training: dict[str, Any], out: Path) -> Path | None:
    curves = training.get("curves", {})
    train = curves.get("train", [])
    if not train:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    steps = [p["step"] for p in train if p.get("loss") is not None]
    losses = [p["loss"] for p in train if p.get("loss") is not None]
    ax.plot(steps, losses, color=OKABE_ITO[0], linewidth=1.2, label="train loss")
    lrs = [p.get("learning_rate") for p in train if p.get("learning_rate") is not None]
    if len(lrs) == len(steps):
        ax2 = ax.twinx()
        ax2.plot(steps, lrs, color=OKABE_ITO[1], linestyle="--", linewidth=1.0,
                 label="learning rate")
        ax2.set_ylabel("learning rate")
        ax2.grid(False)
    evals = curves.get("eval", [])
    if evals:
        ax.plot([p["step"] for p in evals], [p["eval_loss"] for p in evals], "o-",
                color=OKABE_ITO[2], markersize=3, linewidth=1.2, label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(
        f"LoRA training · {training.get('seed')} seed · "
        f"{(training.get('precision') or {}).get('label', '')}"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, out)


def plot_complexity(comparison: dict[str, Any], out: Path) -> Path | None:
    runs = comparison.get("runs", {})
    labels = [label for label in RUN_ORDER if label in runs]
    buckets = sorted({b for label in labels for b in runs[label].get("by_bucket", {})})
    if not buckets:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    width = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        values = [
            runs[label].get("by_bucket", {}).get(b, {}).get("execution_match", {}).get("value", 0)
            for b in buckets
        ]
        positions = [j + i * width - 0.4 + width / 2 for j in range(len(buckets))]
        ax.bar(positions, values, width=width, color=OKABE_ITO[i], label=LABEL_TEXT.get(label, label))
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f"{b}\n(n={runs[labels[0]]['by_bucket'][b]['n']})" if labels else b
                        for b in buckets])
    ax.set_ylabel("execution match")
    ax.set_ylim(0, 1)
    ax.set_title("Execution match by gold-query complexity")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, out)


def plot_failure_taxonomy(error_analysis: dict[str, Any], out: Path) -> Path | None:
    taxonomy = error_analysis.get("taxonomy", {})
    if not taxonomy:
        return None
    labels = [label for label in RUN_ORDER if label in taxonomy]
    categories = sorted({c for label in labels for c in taxonomy[label]})
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    width = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        values = [taxonomy[label].get(c, 0) for c in categories]
        positions = [j + i * width - 0.4 + width / 2 for j in range(len(categories))]
        ax.barh(positions, values, height=width, color=OKABE_ITO[i], label=LABEL_TEXT.get(label, label))
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("count")
    ax.set_title("Failure taxonomy (counts on the test split)")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out)


def plot_forgetting(forgetting: dict[str, Any], out: Path) -> Path | None:
    delta = forgetting.get("delta")
    if not delta:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ppl = delta["wikitext_perplexity"]
    axes[0].bar(["before", "after"], [ppl["before"], ppl["after"]], color=[OKABE_ITO[0], OKABE_ITO[3]])
    axes[0].set_title("WikiText-2 perplexity (lower is better)")
    arc = delta["arc_easy_accuracy"]
    axes[1].bar(["before", "after"], [arc["before"], arc["after"]], color=[OKABE_ITO[0], OKABE_ITO[3]])
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"ARC-Easy accuracy (n={arc['n']})")
    axes[0].set_ylabel("perplexity")
    axes[1].set_ylabel("accuracy")
    return _save(fig, out)


def plot_ablations(ablations: dict[str, Any], out: Path) -> Path | None:
    points = ablations.get("points", [])
    if not points:
        return None
    grids: dict[str, list[dict[str, Any]]] = {}
    for p in points:
        grids.setdefault(p["grid"], []).append(p)
    fig, axes = plt.subplots(1, len(grids), figsize=(3.0 * len(grids), 2.8), squeeze=False)
    for ax, (grid, rows) in zip(axes[0], grids.items(), strict=False):
        labels = [str(r["value"]) for r in rows]
        values = [r["val"]["execution_match"]["value"] for r in rows]
        ax.bar(range(len(rows)), values, color=OKABE_ITO[0])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(grid.replace("_", " "), fontsize=9)
        ax.set_ylim(0, max(0.2, max(values) * 1.25 if values else 0.2))
    axes[0][0].set_ylabel("val execution match")
    fig.suptitle("Ablations (selected on validation)", fontsize=10)
    return _save(fig, out)


def plot_learning_curve(ablations: dict[str, Any], out: Path) -> Path | None:
    rows = [p for p in ablations.get("points", []) if p["grid"] == "train_size"]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.get("train_examples") or 0)
    xs = [r.get("train_examples") or 0 for r in rows]
    ys = [r["val"]["execution_match"]["value"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(xs, ys, "o-", color=OKABE_ITO[0], markersize=4)
    ax.set_xlabel("training examples")
    ax.set_ylabel("val execution match")
    ax.set_title("Learning curve vs data size")
    return _save(fig, out)


def plot_efficiency(efficiency: dict[str, Any], out: Path) -> Path | None:
    latency = efficiency.get("latency", {})
    if not latency:
        return None
    labels = list(latency)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    axes[0].bar(labels, [latency[run_label].get("p50_ms", 0) for run_label in labels],
                yerr=[latency[run_label].get("p95_ms", 0) - latency[run_label].get("p50_ms", 0) for run_label in labels],
                color=OKABE_ITO[: len(labels)], capsize=3)
    axes[0].set_title("Latency (p50 with p95 whisker)")
    axes[0].set_ylabel("ms")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, [latency[run_label].get("tokens_per_second", 0) or 0 for run_label in labels],
                color=OKABE_ITO[: len(labels)])
    axes[1].set_title("Tokens per second")
    axes[1].tick_params(axis="x", rotation=15)
    return _save(fig, out)


def make_all_figures(cfg: Config, data_dir: Path | None = None) -> list[Path]:
    data_dir = data_dir or repo_root() / "web" / "public" / "data"
    out_dir = cfg.figures_dir
    _style()
    comparison = _load(data_dir / "comparison.json")
    training = _load(data_dir / "training.json")
    ablations = _load(data_dir / "ablations.json")
    forgetting = _load(data_dir / "forgetting.json")
    errors = _load(data_dir / "error_analysis.json")
    efficiency = _load(data_dir / "efficiency.json")
    made = []
    for path in (
        plot_metrics_comparison(comparison, out_dir / "metrics_comparison.png"),
        plot_training_curves(training, out_dir / "training_curves.png"),
        plot_complexity(comparison, out_dir / "complexity_buckets.png"),
        plot_failure_taxonomy(errors, out_dir / "failure_taxonomy.png"),
        plot_forgetting(forgetting, out_dir / "forgetting.png"),
        plot_ablations(ablations, out_dir / "ablations.png"),
        plot_learning_curve(ablations, out_dir / "learning_curve.png"),
        plot_efficiency(efficiency, out_dir / "efficiency.png"),
    ):
        if path is not None:
            made.append(path)
    log.info("generated %d figures", len(made))
    return made
