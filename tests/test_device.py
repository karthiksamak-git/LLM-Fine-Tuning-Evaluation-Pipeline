"""Precision policy: every branch of the pure function is covered."""

from __future__ import annotations

import pytest

from mlfte.device import select_precision


def test_cuda_bf16_uses_bf16_without_scaler():
    spec = select_precision("cuda", True)
    assert spec.dtype_name == "bfloat16"
    assert spec.use_autocast is True
    assert spec.use_grad_scaler is False
    assert spec.autocast_dtype_name == "bfloat16"
    assert spec.mixed_precision is True


def test_cuda_without_bf16_uses_fp16_with_scaler():
    spec = select_precision("cuda", False)
    assert spec.dtype_name == "float16"
    assert spec.use_autocast is True
    assert spec.use_grad_scaler is True
    assert spec.autocast_dtype_name == "float16"
    assert "GradScaler" in spec.label


def test_mps_is_fp32_without_mixed_precision():
    spec = select_precision("mps", False)
    assert spec.dtype_name == "float32"
    assert spec.use_autocast is False
    assert spec.use_grad_scaler is False
    assert "not used" in spec.reason


def test_cpu_is_fp32_and_says_mixed_precision_not_used():
    spec = select_precision("cpu", False)
    assert spec.dtype_name == "float32"
    assert spec.mixed_precision is False
    assert spec.reason.startswith("mixed precision: not used")


def test_torch_dtype_property_resolves():
    import torch

    assert select_precision("cpu", False).dtype is torch.float32
    assert select_precision("cuda", True).dtype is torch.bfloat16


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        select_precision("tpu", False)


def test_to_dict_is_json_serialisable():
    payload = select_precision("cpu", False).to_dict()
    assert payload["mixed_precision"] is False
    assert isinstance(payload["label"], str)
