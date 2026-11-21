"""Shared fixtures: a tiny randomly initialised causal LM and the real tokenizer."""

from __future__ import annotations

import pytest
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session")
def tokenizer():
    from mlfte.data.tokenize import load_tokenizer

    try:
        return load_tokenizer(MODEL_NAME)
    except Exception as exc:  # pragma: no cover - offline CI
        pytest.skip(f"tokenizer unavailable (offline?): {exc}")


@pytest.fixture
def tiny_model(tokenizer):
    """A ~1M parameter randomly initialised Llama model with the real vocab size."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    return model
