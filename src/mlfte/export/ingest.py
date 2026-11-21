"""Ingest results exported on a GPU notebook (``mlfte ingest <zip>``).

The zip must contain ``results/export/*.json`` produced by ``mlfte export`` (or the
``export/`` folder of a Colab run). Every payload is schema-validated and the claimed
profile is checked against the hardware recorded in ``env.json`` before anything is copied
into the dashboard.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from mlfte.export.web_artifacts import (
    ARTIFACT_FILES,
    validate_payload,
)
from mlfte.logging_utils import get_logger

log = get_logger(__name__)

PROFILE_DEVICE = {"cpu": "cpu", "gpu": "cuda", "smoke": "cpu"}


def _safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = dest / member
            if not str(target.resolve()).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(dest)


def _find_export_dir(root: Path) -> Path:
    for candidate in root.rglob("summary.json"):
        parent = candidate.parent
        if all((parent / name).exists() for name in ARTIFACT_FILES):
            return parent
    raise FileNotFoundError(
        f"no directory with all of {list(ARTIFACT_FILES)} found inside the zip"
    )


def _check_hardware_claim(payload: dict[str, Any], expect_profile: str) -> None:
    profile = payload.get("profile")
    if profile != expect_profile:
        raise ValueError(f"zip reports profile {profile!r}, expected {expect_profile!r}")
    env = payload.get("env") or {}
    hardware = env.get("hardware") or {}
    device = hardware.get("device")
    expected_device = PROFILE_DEVICE.get(profile)
    if expected_device and device and device != expected_device:
        raise ValueError(
            f"profile {profile!r} claims device {expected_device!r} but env.json says {device!r}"
        )
    if profile == "gpu" and not hardware.get("gpu_available"):
        raise ValueError("profile is 'gpu' but env.json reports no GPU")
    if profile in ("cpu", "smoke") and hardware.get("gpu_available"):
        log.warning("profile %s ran on a machine that also has a GPU; recorded as-is", profile)


def ingest(
    zip_path: Path,
    expect_profile: str = "gpu",
    web_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate and install exported artifacts from a notebook run."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    web_data_dir = web_data_dir or Path(__file__).resolve().parents[3] / "web" / "public" / "data"
    web_data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _safe_extract(zip_path, tmp_path)
        export_dir = _find_export_dir(tmp_path)
        payloads: dict[str, dict[str, Any]] = {}
        for name in ARTIFACT_FILES:
            payload = json.loads((export_dir / name).read_text())
            validate_payload(name, payload)
            payloads[name] = payload
        _check_hardware_claim(payloads["env.json"], expect_profile)
        for name, payload in payloads.items():
            (web_data_dir / name).write_text(json.dumps(payload, indent=2) + "\n")

        figures = list(tmp_path.rglob("*.png"))
        copied_figures = []
        if figures:
            fig_dir = Path(__file__).resolve().parents[3] / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            for figure in figures:
                target = fig_dir / figure.name
                shutil.copy2(figure, target)
                copied_figures.append(target.name)

    summary = payloads["summary.json"]
    info = {
        "installed_files": list(payloads),
        "profile": payloads["env.json"].get("profile"),
        "hardware": payloads["summary.json"].get("hardware"),
        "git_commit": payloads["summary.json"].get("git_commit"),
        "figures": copied_figures,
        "headline": summary.get("headline"),
    }
    log.info("ingested %d artifacts from %s", len(payloads), zip_path)
    return info
