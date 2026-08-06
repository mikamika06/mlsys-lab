# JIT Cache Key Signature Classifier

Our distributed kernel dispatch engine experienced unexpected cache misses and redundant JIT re-compilations during production rollouts. Engineers observed that Triton kernels were being recompiled when passing tensors with identical shapes and dtypes, while changing non-constexpr scalar arguments sometimes failed to trigger necessary recompilations or caused spurious cache key lookups.

To stabilize kernel execution overhead and eliminate redundant compilation latency, we need to formalize how arguments participate in the Triton JIT compilation cache key.

Your task is to implement a classification and key-generation library (`tritoncache/cache_key.py`) and a cache manager (`tritoncache/manager.py`) that strictly categorizes kernel signature fields, builds deterministic cache keys, and handles kernel dispatch lookup.

## Milestones

1. **Signature Field Classification & Key Construction**: Implement `classify_arg` and `build_cache_key` in `tritoncache/cache_key.py`. Categorize function signature parameters based on whether they are marked as `tl.constexpr`, tensor pointers, or non-constexpr scalar values. Construct a normalized string key encoding only the participating fields.
2. **Cache Manager & Specialization Rules**: Implement `JITCacheManager` in `tritoncache/manager.py`. Map input argument tuples to their specialized key representations, tracking cache hits vs. cache misses, and handling specialized metadata (e.g. alignment or constant zero/one hints if applicable).
3. **Regression Test Safeguard**: Write `tests/test_regression.py` that verifies cache key correctness and ensures that treating regular dynamic scalar values as `constexpr` key dependencies (or vice versa) is caught as a regression.
