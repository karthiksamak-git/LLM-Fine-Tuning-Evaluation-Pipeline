"""Config loading, overrides and error handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlfte.config import Config, load_ablations, load_config, parse_overrides


@pytest.mark.parametrize("profile", ["cpu", "gpu", "smoke"])
def test_profiles_load(profile):
    cfg = load_config(profile=profile)
    assert cfg.profile == profile
    assert cfg.model.name
    assert cfg.model.max_seq_len > 0


def test_models_are_ungated_defaults():
    assert load_config(profile="cpu").model.name == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert load_config(profile="gpu").model.name == "Qwen/Qwen2.5-0.5B-Instruct"


def test_overrides_apply_dotted_paths():
    cfg = load_config(profile="cpu", overrides=["training.learning_rate=0.001", "lora.r=8"])
    assert cfg.training.learning_rate == 0.001
    assert cfg.lora.r == 8


def test_unknown_override_path_raises():
    with pytest.raises(KeyError):
        load_config(profile="cpu", overrides=["training.nope=1"])


def test_unknown_section_in_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "profile: cpu\nmodel:\n  name: HuggingFaceTB/SmolLM2-135M-Instruct\ntypo_section:\n  a: 1\n"
    )
    with pytest.raises(ValidationError):
        load_config(path=path)


def test_override_types_come_from_yaml():
    assert parse_overrides(["lora.r=8"]) == {"lora.r": 8}
    assert parse_overrides(["lora.dropout=0.1"]) == {"lora.dropout": 0.1}


def test_ablations_file_is_not_a_run_config():
    with pytest.raises(ValueError):
        load_config(path="configs/ablations.yaml")


def test_ablations_grid_has_expected_keys():
    grid = load_ablations()
    for key in ("rank", "learning_rate", "target_modules", "train_size", "loss_masking"):
        assert key in grid["grids"]
    assert grid["budget"]["eval_split"] == "val"


def test_updated_returns_independent_copy():
    cfg = load_config(profile="cpu")
    other = cfg.updated(profile="smoke")
    assert cfg.profile == "cpu"
    assert other.profile == "smoke"
    assert isinstance(other, Config)
