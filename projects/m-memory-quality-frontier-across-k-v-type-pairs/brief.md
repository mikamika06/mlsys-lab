# Ticket #8402: Sudden VRAM Exhaustion and Slowdowns on 12GB Local Inference Nodes

**Reporter**: Infrastructure Team
**Severity**: High
**Environment**: Local inference runtime running llama.cpp quantized model instances on 12 GB GPU targets.

## Symptom Description
We are deploying local LLM endpoints on consumer hardware with 12 GB VRAM budgets. To fit longer prompt context windows, our deployment service configures asymmetric Key/Value cache quantization types—specifically setting Key cache to `q8_0` and Value cache to `q4_0` or `q4_1`—under the assumption that reducing Value bit width will save significant memory without hurting perplexity.

However, during long-context benchmarking, nodes configured with mixed K/V types frequently crash with out-of-memory errors (`CUDA out of memory` / `GGML VRAM allocation failed`) at sequence lengths that theoretically should fit well within the 12 GB budget. Furthermore, when these requests do manage to run, token generation speed drops dramatically compared to standard `q8_0`/`q8_0` or `f16`/`f16` configurations.

Attempts to systematically select optimal (K, V) precision pairs along the memory-quality trade-off curve have produced contradictory results, with certain precision combinations using more runtime memory than expected and failing to scale context capacity as predicted by simple per-element bit counting.
