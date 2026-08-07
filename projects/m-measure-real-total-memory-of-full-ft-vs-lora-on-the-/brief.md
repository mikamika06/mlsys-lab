# Fine-Tuning Memory Account Budget Mismatch and OOM Diagnostic

## Symptom
Our fine-tuning orchestration platform is incorrectly estimating GPU memory allocation during distributed fine-tuning jobs on our benchmark LLM architecture (`TinyLLM-1B`). Job planning reports that switching from full fine-tuning (Full-FT) to Low-Rank Adaptation (LoRA) and 4-bit Quantized LoRA (QLoRA) should drastically alter parameter footprints, but production runs either Out-Of-Memory (OOM) or reserve far too much VRAM because the scheduler relies on inaccurate memory formulas.

Specifically, the platform scheduler's static memory auditor fails to account for:
1. The exact breakdown of base weight storage, optimizer states (AdamW FP32 state vectors), gradients, and activation memory during Full-FT vs LoRA with bf16 base models.
2. The exact footprint savings achieved when swapping a 16-bit base model (bf16, 2 bytes/param) for a 4-bit quantized base model (QLoRA, 0.55 bytes/param including block quantization scale overhead).
3. Verification that trainable parameter counts remain strictly invariant between standard LoRA (bf16 base) and QLoRA (4-bit base) for identical adapter ranks $r$ and targeted projection layers.

When attempting to launch fine-tuning jobs on high-density nodes, the scheduler's estimation drift causes jobs to be misrouted. We need a deterministic memory accounting module (`ftmem`) that calculates exact byte footprints for base parameters, adapter parameters, optimizer states, gradient buffers, and activation peak memory across Full-FT, LoRA (bf16), and QLoRA (4-bit) modes, as well as verifying parameter count invariants.
