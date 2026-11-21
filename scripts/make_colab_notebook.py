"""Generate ``notebooks/run_on_colab.ipynb`` (kept in sync with the CLI)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "run_on_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


CELLS = [
    md(
        """
# LLM fine-tuning + evaluation — GPU profile

Runs the **gpu** profile (`Qwen/Qwen2.5-0.5B-Instruct`, LoRA) end-to-end on a free
Colab/Kaggle GPU, then packages the exported artifacts so you can ingest them locally.

Runtime → Change runtime type → **T4 GPU** (or better, ≥ 8 GB VRAM).

The notebook never touches the dashboard in this repository: it produces a zip whose files
are schema-validated by `mlfte ingest` before they replace `web/public/data/`.
"""
    ),
    md("## 0. Hardware check"),
    code(
        """
!nvidia-smi
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, round(p.total_memory / 1024**3, 2), "GB | bf16:", torch.cuda.is_bf16_supported())
"""
    ),
    md("## 1. Clone the repository\n\nSet `REPO_URL` to your fork/clone."),
    code(
        """
REPO_URL = "https://github.com/OWNER/llm-finetune-eval.git"  # <-- edit me
%cd /content
!rm -rf llm-finetune-eval
!git clone $REPO_URL llm-finetune-eval
%cd llm-finetune-eval
"""
    ),
    md("## 2. Install pinned dependencies"),
    code(
        """
!pip install -q --upgrade pip
!pip install -q -r requirements.txt
!pip install -q -e .
"""
    ),
    md(
        """
## 3. Run the GPU profile

The pipeline picks bf16 autocast when the GPU supports it, otherwise fp16 + GradScaler,
and records the decision in every result file. Expect roughly 60–120 minutes on a T4.
"""
    ),
    code(
        """
import os
os.environ["HF_HOME"] = "/content/hf_cache"
!python -m mlfte.cli all --profile gpu 2>&1 | tee /content/run.log
"""
    ),
    md("## 4. Package the artifacts"),
    code(
        """
import zipfile, pathlib, shutil

export = pathlib.Path("results/export")
figures = pathlib.Path("figures")
out = pathlib.Path("/content/results-gpu.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in export.rglob("*.json"):
        zf.write(path, path.relative_to("results"))
    for path in figures.glob("*.png"):
        zf.write(path, pathlib.Path("figures") / path.name)
print(out, round(out.stat().st_size / 1024**2, 2), "MB")
"""
    ),
    code(
        """
from google.colab import files  # only exists on Colab
files.download("/content/results-gpu.zip")
"""
    ),
    md(
        """
## 5. Ingest locally

Download the zip, then from the repository root:

```bash
python -m mlfte.cli ingest /path/to/results-gpu.zip --expect-profile gpu
python -m mlfte.cli export --profile gpu   # regenerate figures + README tables
```

`ingest` validates every JSON payload against the schema and checks that `env.json`
actually reports a CUDA device before anything replaces `web/public/data/`.
"""
    ),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
