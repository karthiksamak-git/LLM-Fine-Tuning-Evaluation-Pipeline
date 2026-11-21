"""Hand-written dynamic-padding collator (no DataCollatorForSeq2Seq dependency)."""

from __future__ import annotations

from typing import Any

import torch

IGNORE_INDEX = -100


class CausalLMCollator:
    """Right-pad input ids, pad labels with -100 and build the attention mask.

    Works on a list of dicts holding python lists (as returned by an HF Dataset) or
    tensors. Dynamic padding means every batch is only as long as its longest member.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("collator received an empty batch")
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            step = self.pad_to_multiple_of
            max_len = ((max_len + step - 1) // step) * step

        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, feature in enumerate(features):
            ids = torch.as_tensor(feature["input_ids"], dtype=torch.long)
            n = ids.numel()
            if n > max_len:
                raise ValueError(f"sequence of length {n} exceeds batch length {max_len}")
            input_ids[i, :n] = ids
            attention_mask[i, :n] = torch.as_tensor(
                feature.get("attention_mask", [1] * n), dtype=torch.long
            )
            labels[i, :n] = torch.as_tensor(feature["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
