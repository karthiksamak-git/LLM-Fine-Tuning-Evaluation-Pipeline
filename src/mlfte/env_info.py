"""Environment metadata attached to every result file."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from mlfte.config import Config, repo_root
from mlfte.device import hardware_snapshot

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "numpy",
    "pandas",
    "scipy",
    "sqlglot",
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return None


def collect_env(cfg: Config | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full environment record: python, packages, hardware, precision, git, timestamp."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "hardware": hardware_snapshot(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if cfg is not None:
        info["profile"] = cfg.profile
        info["config"] = cfg.to_dict()
    if extra:
        info.update(extra)
    return info


def data_fingerprint(paths: list[Path]) -> str:
    """Stable sha256 over split files (names + contents), used to tie results to data."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_badge(env: dict[str, Any]) -> dict[str, Any]:
    """The compact profile/hardware badge shown in the dashboard."""
    hw = env.get("hardware", {})
    return {
        "profile": env.get("profile"),
        "device": hw.get("device"),
        "gpu_name": hw.get("gpu_name"),
        "vram_gb": hw.get("vram_gb"),
        "ram_total_gb": hw.get("ram_total_gb"),
        "cpu_count_logical": hw.get("cpu_count_logical"),
        "precision_label": (hw.get("precision") or {}).get("label"),
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
