# Optimize Attention KV Memory and Memory Footprint Transitions

During peak load in production inference, the serving infrastructure is running out of VRAM far earlier than predicted by head count estimations. An audit reveals that our memory budget calculations assume standard multi-head attention (MHA), yet several models are configured for grouped-query attention (GQA) or multi-query attention (MQA). Furthermore, when expanding GQA keys and values to match query head counts during runtime, memory usage surges unexpectedly, triggering out-of-memory (OOM) crashes during long-sequence generation.

We need a unified KV memory accounting module and attention classifier to analyze model configurations, measure exact KV bytes saved by GQA, and prevent memory inflation when expanding KV tensors for tensor-parallel or kernel execution.

Your task:
1. Implement an attention pattern classifier that analyzes model configs (`num_attention_heads`, `num_key_value_heads`) to classify the mechanism as `MHA`, `GQA`, or `MQA`, and computes the grouping ratio.
2. Calculate the exact KV cache memory footprint (in bytes) across different sequence lengths and batch sizes, quantifying the memory bytes saved when storing native GQA caches vs expanded MHA-like projections.
3. Build a CUDA/GPU execution planner that compares native grouped layout storage against runtime expanded KV memory allocations, ensuring memory savings are preserved.
4. Write safety unit tests in `tests/test_regression.py` that verify layout invariance and detect OOM hazards when KV expansion is incorrectly applied at storage time.
