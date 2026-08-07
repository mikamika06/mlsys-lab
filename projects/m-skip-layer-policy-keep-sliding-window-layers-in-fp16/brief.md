When quantizing the KV cache to FP8, we typically accept a small perplexity degradation in exchange for massive memory savings. However, in models with hybrid attention patterns (where some layers compute full attention and others rely on a limited sliding window), blindly applying FP8 across the board is suboptimal. Sliding-window layers naturally have a minimal footprint in the cache because their capacity is capped by the window size. If we quantize these sensitive sliding-window layers to FP8, we take an unnecessary accuracy hit for almost zero memory savings.

Your task is to implement a **skip-layer policy** that preserves sliding-window layers in `float16` and aggressively quantizes only the full-attention layers to `float8`.

You will need to:
1. Implement `assign_kv_dtypes` in `policy/quant.py` to route full layers to `float8` and sliding layers to `float16`.
2. Implement `compute_kv_bytes` to measure the memory cost of a hybrid sequence plan.
3. Write a deterministic numpy-based simulator in `policy/eval.py` that computes the relative error (as a proxy for perplexity delta) of a given dtype assignment against an ideal unquantized baseline. You will inject 5% relative noise for `float8` and 0.1% for `float16`.
4. Add regression tests in `tests/test_regression.py` to enforce that sliding-window layers are never inadvertently downgraded to `float8`.
