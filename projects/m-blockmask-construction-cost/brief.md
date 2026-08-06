# Ticket: FlexAttention Causal Overhead and Mask Caching Strategy

## Symptom
During training benchmark runs, switching our long-context transformer backbone from Standard FlashAttention-2 (FA2) to FlexAttention with custom block masking resulted in a unexpected 3.5x wall-clock latency spike on standard causal masking configurations. Profiling indicates that while the attention kernel execution itself is fast, the overhead associated with constructing PyTorch's `BlockMask` on every forward pass dominates the step time.

Furthermore, naive caching of the constructed `BlockMask` across iterations led to memory leaks and silent correctness bugs when sequence lengths varied dynamically between micro-batches.

## Task
We need a robust mask management and caching module to safely drop in FlexAttention without regressing causal training latency or incurring unamortized construction costs.

1. Implement a precise cost and overhead profiler comparing FlexAttention `BlockMask` instantiation costs against standard FlashAttention-2 causal execution paths.
2. Build an amortized `BlockMask` caching policy (`MaskCache`) that dynamically manages mask reuse based on tensor shape, block size, device, and mask functional signature while evicting stale entries safely.
3. Add a safeguard regression suite in `tests/test_regression.py` that verifies the mask caching policy invalidates cached masks when sequence lengths change, ensuring stale masks are never reused across different dynamic shapes.
