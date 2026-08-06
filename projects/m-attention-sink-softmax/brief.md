# Ticket: Production generation model outputs diverge and degrade during long context inference

## Symptom Report
Our streaming inference engine for long-context language modeling shows severe output corruption when running extended generation passes beyond 8k tokens with sliding-window attention enabled. Specifically, when benchmarking generation output against standard full-causal attention across a 32,768 token sequence, token logits begin to drift dramatically after position 2,048, leading to repetitive loops and garbled output.

Curiously, short context evaluations (under 1,024 tokens) match standard baseline outputs within numerical precision. However, during continuous multi-step generation window sweeps, the accumulated relative error of the layer hidden states scales rapidly. Disabling the attention window fixes the generation quality but causes GPU memory consumption to explode past physical VRAM capacity.

We need a robust sliding-window attention kernel module that incorporates dedicated attention-sink keys, maintaining bounded relative error against full attention over 32k sequences while keeping memory footprint clamped to the window budget.

## Requirements
1. Implement sink-augmented sliding window softmax attention in `attnsink/sink_softmax.py`.
2. Implement window sweep output drift profiling utilities in `attnsink/drift.py` to quantify error growth across long contexts up to 32,768 tokens.
3. Add regression tests in `tests/test_regression.py` that verify mathematical invariants and prevent subtle index overlap bugs when sink tokens overlap with the active sliding window.
