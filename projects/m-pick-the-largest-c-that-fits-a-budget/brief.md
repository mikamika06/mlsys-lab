# Context Budget and Quantized KV Cache Sizing

Our llama.cpp inference node is OOMing during long-context batch serving or failing to utilize available VRAM efficiently when adjusting the context size parameter (`-c`). Additionally, users report degraded prefill throughput and unexpected latency spikes when using mixed-precision KV cache quantization types (such as Q4_0 for Keys and Q8_0 for Values).

To fix this, we need an automated cache budget solver that determines the optimal maximum context size `-c` given strict memory constraints. We must also enforce FlashAttention dependencies when quantized KV caches are enabled, and account for fused kernel overheads when Key and Value quantization types are asymmetric.

## Goal

Implement the cache context budget calculator and fused path analyzer in `kvquant/planner.py`:

1. `fit_context_budget(model_config, memory_budget_bytes, kv_type_k, kv_type_v)`: Calculate the maximum context size `-c` (as a multiple of block size 32) that fits strictly within `memory_budget_bytes`. If even a single block of context cannot fit alongside the model weights and fixed overhead, return 0.
2. `check_flash_attn_requirement(kv_type_k, kv_type_v, use_flash_attn)`: Validate that FlashAttention is enabled when either Key or Value cache uses quantization (non-f16/f32 formats). Quantized KV caches require fused FlashAttention kernels for dequantization during attention computation.
3. `measure_fused_path_penalty(kv_type_k, kv_type_v)`: Compute the dispatch and execution latency penalty ratio introduced by asymmetric K/V types (e.g., Q4_0 K with Q8_0 V) compared to symmetric K/V types in the fused attention path.

Ensure your implementation passes all harness checks and write unit tests in `tests/test_regression.py` that catch regression faults, such as improper handling of asymmetric K/V types or improper alignment of the `-c` context parameter.
