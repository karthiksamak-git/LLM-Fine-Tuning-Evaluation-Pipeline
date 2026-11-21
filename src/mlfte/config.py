"""Typed configuration (pydantic v2) with YAML loading and dotted CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROFILES = ("cpu", "gpu", "smoke")


def repo_root() -> Path:
    """Repository root (two levels above this module: ``src/mlfte/config.py``)."""
    return Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunConfig(_Base):
    seed: int = 42
    train_seeds: list[int] = Field(default_factory=lambda: [42])
    tag: str = "run"


class ModelConfig(_Base):
    name: str
    max_seq_len: int = 320
    load_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"


class DataConfig(_Base):
    dataset: str = "b-mc2/sql-create-context"
    train_size: int = 1200
    val_size: int = 200
    test_size: int = 300
    min_answer_chars: int = 5
    max_answer_chars: int = 512
    max_prompt_chars: int = 2400


class LoraConfigYaml(_Base):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | Literal["auto", "attention", "all_linear"] = "auto"


class TrainingConfig(_Base):
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = False
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 1
    loss_masking: Literal["completion", "full"] = "completion"


class EvalConfig(_Base):
    max_new_tokens: int = 96
    batch_size: int = 8
    bootstrap_resamples: int = 1000
    exec_timeout_s: float = 2.0
    few_shot_k: int = 3
    arc_subset: int = 100
    wikitext_slice: int = 24


class EfficiencyConfig(_Base):
    latency_repeats: int = 10
    batch_size: int = 1


class Config(_Base):
    profile: Literal["cpu", "gpu", "smoke"] = "cpu"
    run: RunConfig = Field(default_factory=RunConfig)
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoraConfigYaml = Field(default_factory=LoraConfigYaml)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, v: str) -> str:
        if v not in PROFILES:
            raise ValueError(f"unknown profile {v!r}; expected one of {PROFILES}")
        return v

    # ---------- paths ----------
    @property
    def results_dir(self) -> Path:
        return repo_root() / "results" / self.profile

    @property
    def artifacts_dir(self) -> Path:
        return repo_root() / "results" / "export"

    @property
    def outputs_dir(self) -> Path:
        return repo_root() / "outputs" / self.profile

    @property
    def figures_dir(self) -> Path:
        return repo_root() / "figures"

    # ---------- helpers ----------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def updated(self, **sections: Any) -> Config:
        """Return a deep copy with whole top-level sections replaced."""
        data = copy.deepcopy(self.model_dump(mode="python"))
        for key, value in sections.items():
            if key not in data:
                raise KeyError(f"unknown config section {key!r}")
            data[key] = value
        return Config.model_validate(data)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"override path {dotted!r}: {part!r} is not a config section")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"override path {dotted!r}: unknown key {parts[-1]!r}")
    node[parts[-1]] = value


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ``["training.learning_rate=1e-4", "lora.r=8"]`` into nested values."""
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.subkey=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        value = yaml.safe_load(raw)
        parsed[key] = value
    return parsed


def load_config(
    path: str | Path | None = None,
    profile: str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load a YAML config, optionally by profile name, then apply dotted overrides."""
    if path is None:
        if profile is None:
            profile = "cpu"
        path = repo_root() / "configs" / f"{profile}.yaml"
    path = Path(path)
    if not path.is_absolute():
        candidate = repo_root() / path
        path = candidate if candidate.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # `ablations.yaml` is a grid file, not a run config; refuse early with a clear message.
    if "grids" in raw:
        raise ValueError(f"{path.name} is an ablation grid, not a run config")
    cfg = Config.model_validate(raw)
    if profile is not None and cfg.profile != profile:
        cfg = cfg.updated(profile=profile)
    if overrides:
        data = cfg.model_dump(mode="python")
        for dotted, value in parse_overrides(overrides).items():
            _set_dotted(data, dotted, value)
        cfg = Config.model_validate(data)
    return cfg


def load_ablations() -> dict[str, Any]:
    """Load the ablation grid file."""
    path = repo_root() / "configs" / "ablations.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
