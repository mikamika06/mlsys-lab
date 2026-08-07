# Ticket #8042: High Memory Consumption and Out-Of-Memory Errors During Long-Context Generation

## Symptom
Production serving nodes hosting our hybrid-architecture transformer model (with interleaved sliding window attention and global attention layers) are experiencing severe GPU memory exhaustion during long-context prefill and generation. When processing prompts exceeding 4,000 tokens, peak KV cache memory consumption grows uniformly across all model layers.

Monitoring metrics reveal that even though several transformer layers are configured with a localized sliding window attention pattern (e.g., window size of 256 or 512 tokens), the runtime KV cache allocator allocates full-length sequence buffers across every layer in the network. On a 32-layer model where half the layers use sliding windows, memory footprint scales linearly with total context length $T$ across all 32 layers rather than bounding memory growth on windowed layers to $W$.

As a result, maximum achievable concurrent batch size drops sharply on long sequences, and worker processes frequently crash with CUDA Out-Of-Memory (OOM) exceptions.

## Goal
Implement a hybrid KV cache manager that enforces per-layer retention policies, ensuring sliding window layers cap stored keys and values to their declared window size while global layers maintain full context. Verify numerical equivalence of attention outputs and validate VRAM savings under production context lengths.
