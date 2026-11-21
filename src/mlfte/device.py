"""Device and mixed-precision selection.

``select_precision`` is a pure function of ``(device_type, bf16_supported)`` so every
branch is unit-testable without a GPU.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

DEVICE_TYPES = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class PrecisionSpec:
    """The precision policy for one training/inference run."""

    dtype_name: str
    use_grad_scaler: bool
    use_autocast: bool
    autocast_dtype_name: str | None
    label: str
    reason: str

    @property
    def dtype(self) -> Any:
        import torch

        return getattr(torch, self.dtype_name)

    @property
    def autocast_dtype(self) -> Any:
        import torch

        return getattr(torch, self.autocast_dtype_name) if self.autocast_dtype_name else None

    @property
    def mixed_precision(self) -> bool:
        return self.use_autocast

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mixed_precision"] = self.mixed_precision
        return data


def select_precision(device_type: str, bf16_supported: bool) -> PrecisionSpec:
    """Choose (dtype, grad-scaler, autocast) for a device.

    Rules:
      * CUDA with bf16 support  -> bf16 autocast, no scaler (bf16 has fp32-range exponent).
      * CUDA without bf16 (T4)  -> fp16 autocast + GradScaler; LoRA params are cast to fp32
        by ``training.lora.cast_trainable_to_fp32`` to avoid FP16 unscale errors.
      * MPS                     -> fp32, mixed precision not used (unstable autocast ops).
      * CPU                     -> fp32, mixed precision not used (no bf16/fp16 kernels worth it).
    """
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"unknown device type {device_type!r}; expected one of {DEVICE_TYPES}")

    if device_type == "cuda" and bf16_supported:
        return PrecisionSpec(
            dtype_name="bfloat16",
            use_grad_scaler=False,
            use_autocast=True,
            autocast_dtype_name="bfloat16",
            label="bf16 autocast",
            reason="CUDA device reports bfloat16 support; bf16 autocast without a grad scaler.",
        )
    if device_type == "cuda":
        return PrecisionSpec(
            dtype_name="float16",
            use_grad_scaler=True,
            use_autocast=True,
            autocast_dtype_name="float16",
            label="fp16 autocast + GradScaler",
            reason=(
                "CUDA device without bfloat16 support; fp16 autocast with GradScaler and "
                "fp32 trainable (LoRA) parameters."
            ),
        )
    if device_type == "mps":
        return PrecisionSpec(
            dtype_name="float32",
            use_grad_scaler=False,
            use_autocast=False,
            autocast_dtype_name=None,
            label="fp32",
            reason="mixed precision: not used (MPS autocast is not reliable; running fp32).",
        )
    return PrecisionSpec(
        dtype_name="float32",
        use_grad_scaler=False,
        use_autocast=False,
        autocast_dtype_name=None,
        label="fp32",
        reason="mixed precision: not used (CPU device: no fp16/bf16 training benefit).",
    )


def resolve_device(prefer: str | None = None) -> str:
    """Detect the best available device: cuda > mps > cpu."""
    import torch

    available = []
    if torch.cuda.is_available():
        available.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        available.append("mps")
    available.append("cpu")
    if prefer and prefer in available:
        return prefer
    return available[0]


def cuda_bf16_supported() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - very old torch/CUDA combos
        return False


def gpu_info() -> dict[str, Any]:
    """GPU name, total VRAM and capability, or an explicit "no GPU" record."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu_available": False, "gpu_name": None, "vram_gb": None, "capability": None}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_available": True,
        "gpu_name": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "capability": f"{props.major}.{props.minor}",
        "device_count": torch.cuda.device_count(),
    }


def nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return None


def hardware_snapshot() -> dict[str, Any]:
    """Everything needed to honestly describe where a run happened."""
    import os

    import psutil

    device = resolve_device()
    precision = select_precision(device, cuda_bf16_supported())
    info = {
        "device": device,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "nvidia_smi": nvidia_smi(),
        **gpu_info(),
    }
    info["precision"] = precision.to_dict()
    return info
