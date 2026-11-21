"""Command line interface: ``mlfte prepare|baseline|train|eval|ablate|efficiency|analyze|export|ingest|all``."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

from mlfte.config import Config, load_ablations, load_config
from mlfte.data.prepare import prepare, read_split, split_paths
from mlfte.env_info import collect_env, read_json, write_json
from mlfte.logging_utils import get_logger
from mlfte.seed import seed_everything, seeded_generator
from mlfte.tasks import get_task
from mlfte.tasks.base import Example, Task

log = get_logger(__name__)

BASELINE_LABELS = ("base_zero_shot", "base_few_shot")
FEW_SHOT_SEED = 7777


# --------------------------------------------------------------------------- helpers
def _cfg(args: argparse.Namespace) -> Config:
    return load_config(path=args.config, profile=args.profile, overrides=args.override or [])


def _ensure_splits(cfg: Config, task: Task) -> dict[str, Any]:
    return prepare(cfg, task, force=False)


def _test_examples(cfg: Config) -> list[Example]:
    return read_split(split_paths(cfg)["test"])


def _val_examples(cfg: Config) -> list[Example]:
    return read_split(split_paths(cfg)["val"])


def few_shot_examples(cfg: Config, task: Task) -> list[Example]:
    """Fixed 3 exemplars from train: one per complexity bucket, from a fixed seed."""
    train = read_split(split_paths(cfg)["train"])
    rng = seeded_generator(FEW_SHOT_SEED)
    order = rng.permutation(len(train))
    picked: dict[str, Example] = {}
    for index in order:
        example = train[int(index)]
        bucket = task.bucket(example)
        if bucket not in picked:
            picked[bucket] = example
        if len(picked) >= cfg.eval.few_shot_k:
            break
    chosen = list(picked.values())
    payload = {
        "seed": FEW_SHOT_SEED,
        "k": len(chosen),
        "policy": "one exemplar per complexity bucket, first hit in a seeded shuffle of train",
        "examples": [ex.to_dict() for ex in chosen],
        "ids": [ex.id for ex in chosen],
    }
    write_json(cfg.results_dir / "few_shot.json", payload)
    return chosen


def _eval_paths(cfg: Config, label: str) -> dict[str, Path]:
    base = cfg.results_dir / "eval"
    return {
        "generations": base / f"{label}.jsonl",
        "metrics": base / f"{label}_metrics.json",
        "items": base / f"{label}_items.jsonl",
    }


def _free(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- commands
def cmd_prepare(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    task = get_task()
    card = prepare(cfg, task, force=args.force)
    print(
        "prepared splits: "
        + ", ".join(f"{name}={len(read_split(path))}" for name, path in split_paths(cfg).items())
    )
    print(f"data fingerprint: {card['data_fingerprint']}")
    tok = card.get("token_stats", {})
    for split in ("train", "val", "test"):
        stats = tok.get(split, {})
        print(
            f"  {split}: truncation {stats.get('truncation_fraction')}, "
            f"dropped {stats.get('dropped')}"
        )
    return card


def evaluate(
    cfg: Config,
    task: Task,
    examples: list[Example],
    label: str,
    adapter_path: str | Path | None = None,
    merged_base: str | Path | None = None,
    few_shot: list[Example] | None = None,
    force: bool = False,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate + score one run, caching generations and per-item scores on disk."""
    from mlfte.evaluation.generate import (
        generate_predictions,
        load_generations,
        load_model_for_inference,
    )
    from mlfte.evaluation.metrics import aggregate_items, score_item

    paths = _eval_paths(cfg, label)
    if paths["metrics"].exists() and not force:
        log.info("reusing existing eval for %s", label)
        return read_json(paths["metrics"])

    if not paths["generations"].exists() or force:
        model, tokenizer = load_model_for_inference(
            cfg,
            base_model=str(merged_base) if merged_base else None,
            adapter_path=adapter_path,
            merged=bool(merged_base),
        )
        generate_predictions(
            model,
            tokenizer,
            task,
            examples,
            cfg,
            paths["generations"],
            few_shot=few_shot,
            max_new_tokens=max_new_tokens,
            label=label,
        )
        _free(model)

    generations = {rec["id"]: rec for rec in load_generations(paths["generations"])}
    items = []
    for example in examples:
        record = generations.get(example.id)
        if record is None:
            continue
        item = score_item(task, example, record["generation"], cfg, seed=cfg.run.seed)
        item["generation"] = record["generation"]
        items.append(item)
    with paths["items"].open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = aggregate_items(items, cfg, seed=cfg.run.seed)
    payload = {
        "label": label,
        "metrics": metrics["metrics"],
        "by_bucket": metrics["by_bucket"],
        "n": metrics["n"],
        "gold_execution_failures": metrics["gold_execution_failures"],
        "limitations": metrics["limitations"],
        "decode": {
            "max_new_tokens": max_new_tokens or cfg.eval.max_new_tokens,
            "batch_size": cfg.eval.batch_size,
            "decoding": "greedy",
        },
        "adapter_path": str(adapter_path) if adapter_path else None,
        "merged": bool(merged_base),
        "few_shot_k": len(few_shot or []),
        "env": collect_env(cfg),
    }
    write_json(paths["metrics"], payload)
    log.info("%s: %s", label, {k: round(v["value"], 4) for k, v in payload["metrics"].items()})
    return payload


def cmd_baseline(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.evaluation.general import forgetting_report

    task = get_task()
    _ensure_splits(cfg, task)
    examples = _test_examples(cfg)
    shots = few_shot_examples(cfg, task)
    result = {}
    result["base_zero_shot"] = evaluate(cfg, task, examples, "base_zero_shot", force=args.force)
    result["base_few_shot"] = evaluate(
        cfg, task, examples, "base_few_shot", few_shot=shots, force=args.force
    )
    forgetting_path = cfg.results_dir / "forgetting_before.json"
    if not forgetting_path.exists() or args.force:
        from mlfte.evaluation.generate import load_model_for_inference

        model, tokenizer = load_model_for_inference(cfg)
        report = forgetting_report(model, tokenizer, cfg, label="base")
        write_json(forgetting_path, report)
        _free(model)
    return result


def cmd_train(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.training.train import train

    task = get_task()
    _ensure_splits(cfg, task)
    seed = args.seed if args.seed is not None else cfg.run.train_seeds[0]
    metrics = train(cfg, task, seed=seed, resume_from_checkpoint=args.resume)
    if args.no_merge:
        return metrics
    _merge_and_verify(cfg, seed)
    return metrics


def _merge_and_verify(cfg: Config, seed: int) -> dict[str, Any]:
    from mlfte.training.merge import merge_adapter, verify_parity
    from mlfte.training.train import adapter_dir

    adapter = adapter_dir(cfg, seed)
    merged_dir = cfg.outputs_dir / f"seed{seed}" / "merged"
    parity_path = cfg.results_dir / "merge" / "parity.json"
    if not merged_dir.exists():
        merge_adapter(cfg, adapter, merged_dir)
    parity = verify_parity(cfg, adapter, merged_dir)
    write_json(parity_path, parity)
    return parity


def cmd_eval(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.evaluation.general import forgetting_delta, forgetting_report
    from mlfte.evaluation.generate import load_model_for_inference
    from mlfte.training.train import adapter_dir

    task = get_task()
    _ensure_splits(cfg, task)
    if not _eval_paths(cfg, "base_zero_shot")["metrics"].exists():
        cmd_baseline(cfg, args)
    examples = _test_examples(cfg)
    shots = few_shot_examples(cfg, task)
    seed = args.seed if args.seed is not None else cfg.run.train_seeds[0]
    adapter = adapter_dir(cfg, seed)
    result = {
        "finetuned_adapter": evaluate(
            cfg, task, examples, "finetuned_adapter", adapter_path=adapter, force=args.force
        ),
        "finetuned_merged": evaluate(
            cfg,
            task,
            examples,
            "finetuned_merged",
            merged_base=cfg.outputs_dir / f"seed{seed}" / "merged",
            force=args.force,
        ),
    }
    _ = shots

    forgetting_path = cfg.results_dir / "forgetting.json"
    if not forgetting_path.exists() or args.force:
        before = read_json(cfg.results_dir / "forgetting_before.json")
        model, tokenizer = load_model_for_inference(cfg, adapter_path=adapter)
        after = forgetting_report(model, tokenizer, cfg, label="finetuned_adapter")
        _free(model)
        payload = {
            "before": before,
            "after": after,
            "delta": forgetting_delta(before, after),
            "env": collect_env(cfg),
        }
        write_json(forgetting_path, payload)
    _build_seed_summary(cfg)
    return result


def _build_seed_summary(cfg: Config) -> None:
    from mlfte.evaluation.stats import mean_std

    values = []
    for seed in cfg.run.train_seeds:
        path = cfg.results_dir / "eval" / f"finetuned_adapter_seed{seed}_metrics.json"
        if path.exists():
            values.append(read_json(path)["metrics"]["execution_match"]["value"])
    if not values:
        path = cfg.results_dir / "eval" / "finetuned_adapter_metrics.json"
        if path.exists():
            values.append(read_json(path)["metrics"]["execution_match"]["value"])
    summary = mean_std(values)
    note = (
        f"{len(values)} completed training seed(s). "
        + (
            "The CPU budget only allowed a single seed; seed variance is not estimated."
            if len(values) < 2
            else "Reported as mean ± std over seeds."
        )
    )
    write_json(
        cfg.results_dir / "seeds.json",
        {"execution_match": summary, "n": summary["n"], "seeds": cfg.run.train_seeds, "note": note},
    )


def build_comparison(cfg: Config) -> dict[str, Any]:
    """Paired comparison of the fine-tuned run against every baseline that exists."""
    from mlfte.evaluation.stats import compare_metric

    def items(label: str) -> dict[str, dict[str, Any]]:
        path = _eval_paths(cfg, label)["items"]
        if not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
        return out

    runs = {}
    all_items = {}
    for label in ("base_zero_shot", "base_few_shot", "finetuned_adapter", "finetuned_merged"):
        metrics_path = _eval_paths(cfg, label)["metrics"]
        if not metrics_path.exists():
            continue
        payload = read_json(metrics_path)
        runs[label] = {
            "metrics": payload["metrics"],
            "by_bucket": payload["by_bucket"],
            "n": payload["n"],
            "few_shot_k": payload["few_shot_k"],
            "merged": payload["merged"],
            "decode": payload["decode"],
            "gold_execution_failures": payload.get("gold_execution_failures"),
        }
        all_items[label] = items(label)

    comparisons: dict[str, Any] = {}
    for ft in ("finetuned_adapter", "finetuned_merged"):
        if ft not in all_items:
            continue
        for base in ("base_zero_shot", "base_few_shot"):
            if base not in all_items:
                continue
            common = sorted(set(all_items[ft]) & set(all_items[base]))
            if not common:
                continue
            entry = {"n_paired": len(common)}
            for metric in ("execution_match", "exact_match", "is_valid"):
                new = [bool(all_items[ft][i][metric]) for i in common]
                old = [bool(all_items[base][i][metric]) for i in common]
                entry[metric] = compare_metric(
                    old, new, n_resamples=cfg.eval.bootstrap_resamples, seed=cfg.run.seed
                )
            comparisons[f"{ft}__vs__{base}"] = entry

    card = read_json(cfg.results_dir / "data" / "data_card.json")
    payload = {
        "runs": runs,
        "comparisons": comparisons,
        "test_set": {
            "n": len(_test_examples(cfg)),
            "split_seed": 1234,
            "data_fingerprint": card["data_fingerprint"],
            "policy": card["split"]["policy"],
        },
        "seeds": read_json(cfg.results_dir / "seeds.json") if (cfg.results_dir / "seeds.json").exists() else {},
        "env": collect_env(cfg),
    }
    write_json(cfg.results_dir / "comparison.json", payload)
    log.info("comparison written: %s", list(comparisons))
    return payload


def cmd_ablate(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.evaluation.generate import (
        generate_predictions,
        load_generations,
        load_model_for_inference,
    )
    from mlfte.evaluation.metrics import aggregate_items, score_item
    from mlfte.training.train import adapter_dir, train

    task = get_task()
    _ensure_splits(cfg, task)
    grid = load_ablations()
    max_steps = int(grid["budget"]["max_steps_by_profile"].get(cfg.profile, 40))
    val_all = _val_examples(cfg)
    val_subset = val_all[: min(len(val_all), int(grid["budget"].get("val_examples", 48)))]
    points: list[dict[str, Any]] = []

    def run_point(name: str, value: Any, overrides: dict[str, Any]) -> None:
        tag = f"abl_{name}_{_slug(value)}"
        out_path = cfg.results_dir / "ablations" / f"{tag}.json"
        if out_path.exists() and not args.force:
            points.append(read_json(out_path))
            log.info("reusing ablation %s", tag)
            return
        variant = _variant(cfg, overrides)
        seed = variant.run.train_seeds[0]
        metrics = train(variant, task, seed=seed, tag=tag, max_steps=max_steps)
        adapter = adapter_dir(variant, seed, tag)
        generations_path = cfg.results_dir / "ablations" / f"{tag}.jsonl"
        model, tokenizer = load_model_for_inference(variant, adapter_path=adapter)
        generate_predictions(
            model,
            tokenizer,
            task,
            val_subset,
            variant,
            generations_path,
            max_new_tokens=min(48, variant.eval.max_new_tokens),
            label=tag,
        )
        _free(model)
        gens = {rec["id"]: rec for rec in load_generations(generations_path)}
        items = [
            score_item(task, ex, gens[ex.id]["generation"], variant, seed=seed)
            for ex in val_subset
            if ex.id in gens
        ]
        summary = aggregate_items(items, variant, seed=seed)
        point = {
            "grid": name,
            "value": value,
            "tag": tag,
            "steps": metrics["steps"],
            "wall_clock_s": metrics["wall_clock_s"],
            "train_examples": variant.data.train_size,
            "val": {
                "n": summary["n"],
                "exact_match": summary["metrics"]["exact_match"],
                "is_valid": summary["metrics"]["is_valid"],
                "execution_match": summary["metrics"]["execution_match"],
            },
            "eval_loss": metrics.get("best_eval_loss"),
            "config": {
                "lora_r": variant.lora.r,
                "learning_rate": variant.training.learning_rate,
                "target_modules": variant.lora.target_modules,
                "loss_masking": variant.training.loss_masking,
                "train_size": variant.data.train_size,
            },
        }
        write_json(out_path, point)
        points.append(point)
        log.info("ablation %s=%s -> val exec %s", name, value,
                 point["val"]["execution_match"]["value"])

    grids = grid["grids"]
    if grids["rank"]["enabled"]:
        for r in grids["rank"]["values"]:
            run_point("rank", r, {"lora": {"r": r, "alpha": 2 * r}})
    if grids["learning_rate"]["enabled"]:
        for lr in grids["learning_rate"]["values"]:
            run_point("learning_rate", lr, {"training": {"learning_rate": lr}})
    if grids["target_modules"]["enabled"]:
        for variant in grids["target_modules"]["values"]:
            run_point("target_modules", variant, {"lora": {"target_modules": variant}})
    if grids["train_size"]["enabled"]:
        for fraction in grids["train_size"]["fractions"]:
            if fraction >= 1.0:
                continue
            size = max(16, int(round(cfg.data.train_size * fraction)))
            run_point("train_size", fraction, {"data": {"train_size": size}})
    if grids["loss_masking"]["enabled"]:
        for masking in grids["loss_masking"]["values"]:
            if masking == cfg.training.loss_masking:
                continue
            run_point("loss_masking", masking, {"training": {"loss_masking": masking}})

    selection = _select_ablation(points)
    payload = {
        "profile": cfg.profile,
        "budget": {
            "max_steps": max_steps,
            "val_examples": len(val_subset),
            "note": (
                "Ablation points are budget-scaled: they train for fewer steps than the main "
                "run and are evaluated on a validation subset. Selection uses validation only; "
                "the test split is reported for the main configuration."
            ),
        },
        "points": points,
        "selection": selection,
        "env": collect_env(cfg),
    }
    write_json(cfg.results_dir / "ablations.json", payload)
    return payload


def _variant(cfg: Config, overrides: dict[str, Any]) -> Config:
    data = cfg.to_dict()
    for section, values in overrides.items():
        data[section].update(values)
    return Config.model_validate(data)


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def _select_ablation(points: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for point in points:
        grid = point["grid"]
        score = point["val"]["execution_match"]["value"]
        current = best.get(grid)
        if current is None or score > current["val"]["execution_match"]["value"]:
            best[grid] = point
    summary = {grid: {"value": p["value"], "val_execution_match":
                      p["val"]["execution_match"]["value"]} for grid, p in best.items()}
    rationale = (
        "Best validation execution match per grid: "
        + ", ".join(f"{g}={v['value']} ({100 * v['val_execution_match']:.1f}%)"
                    for g, v in summary.items())
        + ". Ties are broken toward the earlier (smaller) setting. The main run keeps the "
        "profile defaults; these points show sensitivity, not a re-tuned final config, "
        "because the CPU budget allowed only one seed."
    )
    return {"best_per_grid": summary, "rationale": rationale}


def cmd_efficiency(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.evaluation.efficiency import efficiency_report
    from mlfte.training.train import adapter_dir

    seed = args.seed if args.seed is not None else cfg.run.train_seeds[0]
    report = efficiency_report(cfg, adapter_dir(cfg, seed))
    payload = {**report, "env": collect_env(cfg)}
    write_json(cfg.results_dir / "efficiency.json", payload)
    return payload


def cmd_analyze(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.evaluation.error_analysis import analyze

    examples = _test_examples(cfg)
    runs: dict[str, list[dict[str, Any]]] = {}
    for label in ("base_zero_shot", "base_few_shot", "finetuned_adapter", "finetuned_merged"):
        path = _eval_paths(cfg, label)["items"]
        if path.exists():
            runs[label] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if "finetuned_adapter" not in runs:
        raise RuntimeError("analyze needs the fine-tuned run; execute `mlfte eval` first")
    primary = "finetuned_adapter"
    baseline = "base_few_shot" if "base_few_shot" in runs else "base_zero_shot"
    analysis = analyze(examples, runs, primary=primary, baseline=baseline)
    for example in analysis["examples"]:
        if isinstance(example.get("bucket"), set):
            example["bucket"] = sorted(example["bucket"])[0]
    payload = {**analysis, "env": collect_env(cfg)}
    write_json(cfg.results_dir / "error_analysis.json", payload)
    log.info("outcomes: %s", analysis["outcomes"])
    return payload


def cmd_export(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.export.web_artifacts import build_all
    from mlfte.figures import make_all_figures
    from mlfte.report import generate_report

    build_comparison(cfg)
    paths = build_all(cfg)
    make_all_figures(cfg)
    generate_report(cfg)
    return {"artifacts": [str(p) for p in paths.values()]}


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.export.ingest import ingest
    from mlfte.report import generate_report

    if not args.zip:
        raise SystemExit("ingest needs a results zip: `mlfte ingest <path/to/export.zip>`")
    info = ingest(Path(args.zip), expect_profile=args.expect_profile or cfg.profile)
    generate_report(cfg)
    return info


def cmd_all(cfg: Config, args: argparse.Namespace) -> dict[str, Any]:
    from mlfte.training.merge import merge_adapter, verify_parity
    from mlfte.training.train import adapter_dir, train

    seed_everything(cfg.run.seed)
    task = get_task()
    log.info("=== PHASE 1: prepare ===")
    cmd_prepare(cfg, args)

    log.info("=== PHASE 2: baselines ===")
    cmd_baseline(cfg, args)

    log.info("=== PHASE 3: training ===")
    seed = cfg.run.train_seeds[0]
    train(cfg, task, seed=seed)
    adapter = adapter_dir(cfg, seed)
    merged_dir = cfg.outputs_dir / f"seed{seed}" / "merged"
    if not merged_dir.exists() or args.force:
        merge_adapter(cfg, adapter, merged_dir)
    write_json(cfg.results_dir / "merge" / "parity.json", verify_parity(cfg, adapter, merged_dir))

    log.info("=== PHASE 4: evaluation ===")
    cmd_eval(cfg, args)

    log.info("=== PHASE 5: ablations ===")
    try:
        cmd_ablate(cfg, args)
    except Exception as exc:  # ablations must not invalidate the main results
        log.error("ablations failed: %s", exc)
        write_json(cfg.results_dir / "ablations.json", {"points": [], "error": str(exc)})

    log.info("=== PHASE 6: efficiency ===")
    cmd_efficiency(cfg, args)

    log.info("=== PHASE 7: analysis ===")
    cmd_analyze(cfg, args)

    log.info("=== PHASE 8: export ===")
    cmd_export(cfg, args)
    log.info("=== DONE ===")
    return {"ok": True}


# --------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlfte", description=__doc__)
    parser.add_argument("command", choices=[
        "prepare", "baseline", "train", "eval", "ablate", "efficiency", "analyze",
        "export", "ingest", "all",
    ])
    parser.add_argument("target", nargs="?", help="zip path for `ingest`")
    parser.add_argument("--profile", choices=("cpu", "gpu", "smoke"), default=None)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted config override, e.g. training.learning_rate=1e-4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--zip", default=None)
    parser.add_argument("--expect-profile", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target and not args.zip:
        args.zip = args.target
    try:
        cfg = _cfg(args)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    handlers = {
        "prepare": cmd_prepare,
        "baseline": cmd_baseline,
        "train": cmd_train,
        "eval": cmd_eval,
        "ablate": cmd_ablate,
        "efficiency": cmd_efficiency,
        "analyze": cmd_analyze,
        "export": cmd_export,
        "ingest": cmd_ingest,
        "all": cmd_all,
    }
    try:
        handlers[args.command](cfg, args)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as exc:
        log.error("%s failed: %s", args.command, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
