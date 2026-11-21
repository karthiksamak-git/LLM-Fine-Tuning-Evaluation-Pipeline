# PROGRESS (gitignored, phase log)

Machine: CPU-only, **no GPU**, Python 3.12.3, Node 26.13. Hardware varied between runners:
timings were first calibrated on 2 vCPU / 3.0 GB, the final run executed on 8 vCPU / 15.8 GB.
Both are recorded per-artifact (`hardware`), so the badge always shows the machine that ran.
Compute profile decision: **cpu** (no CUDA, no MPS). Model: `HuggingFaceTB/SmolLM2-135M-Instruct`.

| Phase | Status | Gate |
| --- | --- | --- |
| 0 Environment + scaffolding | done | imports OK; tokenizer+model download OK; 1-step forward pass on CPU (loss 3.56, logits [1,30,49152]) |
| 1 Core utilities | done | `tests/test_device.py`, `test_config.py`, `test_seed/env` pass; precision branches covered |
| 2 Data pipeline | done | disjoint splits, masking, collator, deterministic split tests pass |
| 3 Evaluation harness | done | gold-vs-gold 100%, rewrites match, wrong/malformed handled (harness uses synthetic seeded SQLite) |
| 4 Baselines | done | zero-shot + 3-shot metrics with CIs saved |
| 5 LoRA fine-tuning | done | adapter-only trainable, overfit, resume, merge parity tests pass |
| 6 After-eval + comparison | pending | comparison with CIs + McNemar |
| 7 Ablations | pending | val-selected grid points |
| 8 Efficiency | pending | latency/size JSON, fp32-vs-mixed skip reason on CPU |
| 9 Error analysis + report | pending | taxonomy + curated examples |
| 10 Full run + figures + export | pending | 11 valid JSON artifacts + figures |
| 11 Web app | pending | lint/build/test + routes |
| 12 CI + docs | pending | make test / make lint |
| 13 GitHub | blocked | `gh` not authenticated |
| 14 Vercel | blocked | `vercel` CLI missing + not authenticated |

## Budget decisions (recorded honestly)

Measured on this box (2 threads, fp32):

- training step, batch 2 x 256 tokens: **5.6 s/step** (~91 tokens/s)
- generation, batch 8, 200-token prompt, 32 new tokens: **6.45 s** (~40 tokens/s)

Consequences:

- `configs/cpu.yaml`: train 1,200 examples, **300 optimizer steps** (effective batch 4 =>
  one epoch), test split **200** examples (not 300), `max_new_tokens=64`.
- Ablations are budget-scaled to 24 steps and a 48-example validation subset; they are
  selected on validation only.
- One training seed on CPU; the GPU config requests three.

## Notes / fixes made during the build

- transformers 5.23 removed `overwrite_output_dir` and `warmup_ratio`; the code uses
  `warmup_steps` and no `overwrite_output_dir`.
- LoRA target discovery originally included the tied `lm_head`, which broke merge parity
  (PEFT unties the head on merge, max logit diff 3.6e-2). Targets are now restricted to
  attention + MLP projections; a regression test asserts `lm_head` is excluded.
- Execution-match harness initially rejected `INTERSECT`/`EXCEPT` as non-read-only; fixed.

## Final CPU run (recorded numbers)

- Hardware: CPU-only, 8 vCPU, 15.8 GB RAM, fp32, no mixed precision.
- Training: 300 optimizer steps, 1.0 epoch, 1379 s wall clock, 4,884,480 trainable params
  (3.50% of 139,399,488), target modules q/k/v/o/gate/up/down on all layers.
- Loss: train 1.417 -> 0.194; validation 0.1851 -> 0.1705 -> 0.1580 (best at step 300).
- Merge parity: max |logit diff| 8.8e-05 (tolerance 5e-3) — passed.
- Peak RSS during training: 2813 MB.
