"""Auto-generate markdown findings from real, exported numbers only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlfte.config import Config, repo_root
from mlfte.logging_utils import get_logger

log = get_logger(__name__)

BEGIN = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def _ci(metric: dict[str, Any] | None, digits: int = 1) -> str:
    if not metric:
        return "n/a"
    value = metric.get("value")
    lo = metric.get("ci_low")
    hi = metric.get("ci_high")
    if value is None:
        return "n/a"
    if lo is None or hi is None:
        return _pct(value, digits)
    return f"{100 * value:.{digits}f}% [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"


def _p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.0001:
        return "< 0.0001"
    return f"{value:.4f}"


def _hardware_line(data: dict[str, Any]) -> str:
    hw = data.get("hardware", {}) or {}
    device = hw.get("device", "unknown")
    gpu = hw.get("gpu_name")
    vram = hw.get("vram_gb")
    parts = [f"device=`{device}`", f"profile=`{data.get('profile', 'unknown')}`"]
    if gpu:
        parts.append(f"gpu=`{gpu}` ({vram} GB)")
    if hw.get("cpu_count_logical"):
        parts.append(f"cpu={hw['cpu_count_logical']} vCPU")
    if hw.get("ram_total_gb"):
        parts.append(f"ram={hw['ram_total_gb']} GB")
    if hw.get("precision_label"):
        parts.append(f"precision=`{hw['precision_label']}`")
    return " · ".join(parts)


def build_tables(data_dir: Path) -> str:
    summary = _load(data_dir / "summary.json")
    comparison = _load(data_dir / "comparison.json")
    training = _load(data_dir / "training.json")
    ablations = _load(data_dir / "ablations.json")
    forgetting = _load(data_dir / "forgetting.json")
    efficiency = _load(data_dir / "efficiency.json")
    if not summary:
        return f"{BEGIN}\n_No exported results yet. Run `mlfte all --profile <cpu|gpu>`._\n{END}\n"

    lines: list[str] = [BEGIN, ""]
    lines.append(
        f"**Active profile:** `{summary.get('profile')}` · {_hardware_line(summary)} · "
        f"model `{summary.get('model')}` · data fingerprint `{(summary.get('data_fingerprint') or 'n/a')[:12]}`"
    )
    notice = summary.get("profile_notice")
    if notice:
        lines += ["", f"> **Hardware honesty:** {notice}"]
    lines.append("")

    # ---- before/after ----
    runs = comparison.get("runs", {})
    if runs:
        lines += ["### Before / after on the held-out test split", ""]
        lines.append(
            f"Test n={summary.get('headline', {}).get('test_n')} · mean with 95% bootstrap CIs"
        )
        lines += [
            "",
            "| Run | Exact match | SQL valid | Execution match |",
            "| --- | --- | --- | --- |",
        ]
        for label in (
            "base_zero_shot",
            "base_few_shot",
            "finetuned_adapter",
            "finetuned_merged",
        ):
            payload = runs.get(label)
            if not payload:
                continue
            metrics = payload.get("metrics", {})
            lines.append(
                f"| `{label}` | {_ci(metrics.get('exact_match'))} | "
                f"{_ci(metrics.get('is_valid'))} | {_ci(metrics.get('execution_match'))} |"
            )
        lines.append("")

        comparisons = comparison.get("comparisons", {})
        if comparisons:
            lines += [
                "**Paired significance (McNemar) on execution match**",
                "",
                "| Comparison | Δ absolute | Δ relative | p-value | Significant (0.05) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for name, payload in comparisons.items():
                metric = payload.get("execution_match", {})
                mcnemar = metric.get("mcnemar", {})
                delta = metric.get("absolute_difference")
                rel = metric.get("relative_improvement")
                p = mcnemar.get("p_value")
                sig = "yes" if (p is not None and p < 0.05) else "no"
                lines.append(
                    f"| {name.replace('__', ' ')} | "
                    f"{(f'{100 * delta:+.2f} pp' if delta is not None else 'n/a')} | "
                    f"{(f'{100 * rel:+.1f}%' if rel is not None else 'n/a')} | {_p(p)} | {sig} |"
                )
            lines.append("")

    # ---- seed spread ----
    seeds = comparison.get("seeds", {})
    if seeds and seeds.get("n", 0) >= 1:
        lines += [
            f"**Training seeds:** {seeds.get('n')} seed(s) — execution match "
            f"{_pct((seeds.get('execution_match') or {}).get('mean'))} "
            f"± {_pct((seeds.get('execution_match') or {}).get('std'))} (mean ± std). "
            f"{seeds.get('note', '')}",
            "",
        ]

    # ---- training ----
    if training:
        tp = training.get("throughput", {})
        lora = training.get("lora", {})
        lines += [
            "### Training",
            "",
            f"- Steps: **{tp.get('steps')}** · epochs completed: {tp.get('epochs_completed')} · "
            f"wall clock: {tp.get('wall_clock_s')} s",
            f"- Trainable params: **{lora.get('trainable_parameters'):,}** "
            f"({lora.get('trainable_percent'):.3f}% of {lora.get('total_parameters'):,})"
            if lora.get("trainable_parameters")
            else "- Trainable params: n/a",
            f"- Precision: `{(training.get('precision') or {}).get('label')}` — "
            f"{(training.get('precision') or {}).get('reason')}",
            f"- Throughput: {tp.get('train_tokens_per_second')} tokens/s (estimated) · "
            f"peak RSS {training.get('memory', {}).get('peak_rss_mb')} MB",
            "",
        ]

    # ---- ablations ----
    if ablations.get("points"):
        lines += ["### Ablations (selected on validation)", ""]
        lines.append("| Grid | Value | Val execution match | Val exact match | Steps |")
        lines.append("| --- | --- | --- | --- | --- |")
        for point in ablations["points"]:
            val = point.get("val", {})
            lines.append(
                f"| {point.get('grid')} | `{point.get('value')}` | "
                f"{_pct((val.get('execution_match') or {}).get('value'))} | "
                f"{_pct((val.get('exact_match') or {}).get('value'))} | {point.get('steps')} |"
            )
        lines += ["", f"**Selection:** {ablations.get('selection', {}).get('rationale', 'n/a')}", ""]

    # ---- forgetting ----
    if forgetting.get("delta"):
        delta = forgetting["delta"]
        ppl = delta["wikitext_perplexity"]
        arc = delta["arc_easy_accuracy"]
        lines += [
            "### Forgetting checks",
            "",
            "| Check | Before | After | Δ | Direction |",
            "| --- | --- | --- | --- | --- |",
            f"| WikiText-2 perplexity (↓ better) | {ppl['before']:.3f} | {ppl['after']:.3f} | "
            f"{ppl['delta']:+.3f} | {ppl['direction']} |",
            f"| ARC-Easy accuracy (↑ better, n={arc['n']}) | {_pct(arc['before'])} | "
            f"{_pct(arc['after'])} | {100 * arc['delta']:+.2f} pp | {arc['direction']} |",
            "",
        ]

    # ---- efficiency ----
    if efficiency.get("latency"):
        lines += ["### Efficiency", ""]
        lines.append("| Model | p50 latency | p95 latency | tokens/s |")
        lines.append("| --- | --- | --- | --- |")
        for name, lat in efficiency["latency"].items():
            lines.append(
                f"| `{name}` | {lat.get('p50_ms')} ms | {lat.get('p95_ms')} ms | "
                f"{lat.get('tokens_per_second')} |"
            )
        sizes = efficiency.get("sizes", {})
        lines += [
            "",
            f"Adapter size: **{sizes.get('adapter_mb')} MB** · "
            f"trained (LoRA) params {sizes.get('adapter_lora_percent', 0):.3f}% "
            f"of {sizes.get('adapter_total', 0):,} "
            f"({sizes.get('adapter_lora', 0):,} parameters)",
            "",
        ]
        bench = efficiency.get("mixed_precision_benchmark", {})
        if bench.get("skipped"):
            lines += [f"Mixed precision: **{bench.get('reason')}**", ""]
        elif bench:
            lines += [
                "| Precision | seconds | steps/s | peak MB |",
                "| --- | --- | --- | --- |",
                f"| fp32 | {bench.get('fp32', {}).get('seconds')} | "
                f"{bench.get('fp32', {}).get('steps_per_second')} | "
                f"{bench.get('fp32', {}).get('peak_memory_mb')} |",
                f"| {bench.get('precision', {}).get('label')} | {bench.get('mixed', {}).get('seconds')} | "
                f"{bench.get('mixed', {}).get('steps_per_second')} | "
                f"{bench.get('mixed', {}).get('peak_memory_mb')} |",
                "",
            ]

    lines.append(END)
    return "\n".join(lines) + "\n"


def generate_report(cfg: Config, data_dir: Path | None = None) -> Path:
    full = data_dir or repo_root() / "web" / "public" / "data"
    block = build_tables(full)
    report_path = cfg.results_dir / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Results\n\n" + block, encoding="utf-8")

    readme = repo_root() / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if BEGIN in text and END in text:
            head = text.split(BEGIN)[0]
            tail = text.split(END, 1)[1]
            readme.write_text(head + block + tail, encoding="utf-8")
            log.info("updated README results block")
    log.info("wrote %s", report_path)
    return report_path
