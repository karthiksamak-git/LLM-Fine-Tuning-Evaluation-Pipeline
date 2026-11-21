"""Logging helpers: a configured logger, a JSONL writer and a wall-clock timer."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "mlfte", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("mlfte")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger


class JsonlWriter:
    """Append-only JSONL writer that creates parent directories."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, records: Iterable[dict[str, Any]]) -> int:
        n = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                n += 1
        return n

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0
