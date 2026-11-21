"""Decontamination: prove train/val/test are disjoint.

Two checks:
  1. Exact: sha256 over the normalised (question + schema) pair must be unique across splits.
  2. Near-duplicate: 5-gram shingles of the normalised text; any pair with Jaccard >= 0.9
     across two different splits is reported (and fails the gate).
"""

from __future__ import annotations

import hashlib
import re
from itertools import combinations
from typing import Any

from mlfte.tasks.base import Example

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
NEAR_DUP_THRESHOLD = 0.9
SHINGLE_SIZE = 5


def _content(example: Example) -> str:
    return f"{example.question} {example.context}".lower()


def exact_key(example: Example) -> str:
    return hashlib.sha256(_content(example).encode("utf-8")).hexdigest()


def shingles(example: Example, k: int = SHINGLE_SIZE) -> frozenset[int]:
    tokens = _TOKEN_RE.findall(_content(example))
    if len(tokens) < k:
        tokens = tokens or [""]
    grams = {" ".join(tokens[i : i + k]) for i in range(max(1, len(tokens) - k + 1))}
    return frozenset(int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16) for g in grams)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _near_duplicates(
    left: list[Example], right: list[Example], threshold: float = NEAR_DUP_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Pairs (i, j, similarity) above threshold, using an inverted index for candidates."""
    left_sets = [shingles(e) for e in left]
    index: dict[int, list[int]] = {}
    for i, s in enumerate(left_sets):
        for h in s:
            index.setdefault(h, []).append(i)

    # A pair needs at least ceil(threshold * |union|) shared shingles; use a cheap lower bound.
    hits: list[tuple[int, int, float]] = []
    for j, ex in enumerate(right):
        rs = shingles(ex)
        counts: dict[int, int] = {}
        for h in rs:
            for i in index.get(h, ()):  # type: ignore[union-attr]
                counts[i] = counts.get(i, 0) + 1
        for i, shared in counts.items():
            union = len(left_sets[i]) + len(rs) - shared
            if union and shared / union >= threshold:
                hits.append((i, j, shared / union))
    return hits


def decontaminate(
    splits: dict[str, list[Example]], threshold: float = NEAR_DUP_THRESHOLD
) -> dict[str, Any]:
    """Return a report with per-pair overlap counts and the overall disjoint verdict."""
    exact: dict[str, set[str]] = {name: {exact_key(e) for e in exs} for name, exs in splits.items()}
    pairs: dict[str, Any] = {}
    disjoint = True
    for a, b in combinations(splits, 2):
        exact_overlap = len(exact[a] & exact[b])
        near = _near_duplicates(splits[a], splits[b], threshold)
        pairs[f"{a}__{b}"] = {
            "exact_overlap": exact_overlap,
            "near_duplicate_pairs": len(near),
            "near_duplicate_examples": [
                {"i": i, "j": j, "jaccard": round(s, 4), "left_id": splits[a][i].id,
                 "right_id": splits[b][j].id}
                for i, j, s in near[:10]
            ],
        }
        if exact_overlap or near:
            disjoint = False

    duplicate_within: dict[str, int] = {}
    for name, exs in splits.items():
        keys = [exact_key(e) for e in exs]
        duplicate_within[name] = len(keys) - len(set(keys))

    return {
        "method": (
            "sha256 of normalised question+schema for exact duplicates; "
            f"{SHINGLE_SIZE}-gram shingle Jaccard >= {threshold} for near duplicates"
        ),
        "threshold": threshold,
        "pairs": pairs,
        "duplicates_within_split": duplicate_within,
        "disjoint": disjoint,
    }
