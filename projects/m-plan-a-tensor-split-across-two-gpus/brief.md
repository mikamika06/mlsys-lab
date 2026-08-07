# Ticket: Dual-GPU Tensor Split Allocation Failure in Llama.cpp Runtime

## Symptom
When loading large GGUF models across a dual-GPU configuration using the runtime's tensor-split parameters, the initialization phase frequently crashes with out-of-memory (OOM) errors on the primary device (GPU 0) while the secondary device (GPU 1) remains underutilized.

Logs indicate that the current layer distribution assumes a naive uniform distribution of weights across all tensor blocks. However, models featuring varying attention heads, expert routing, or dense projection layers result in highly skewed memory footprints per layer index. Consequently, the statically computed split ratios fail to accurately reflect actual VRAM consumption, causing GPU 0 to exceed its capacity during weight offloading and graph allocation.

We need a dedicated planning module that parses model layer specifications, calculates precise per-layer byte sizes, and determines an optimal tensor split ratio across two GPUs to ensure balanced memory utilization and prevent premature OOM failures during runtime initialization.
