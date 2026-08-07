We are upgrading our transformer inference stack to rely entirely on PyTorch's `scaled_dot_product_attention` (SDPA), but we have run into two significant problems. First, we are seeing out-of-memory errors on large batch sizes because PyTorch is quietly falling back to the slow, memory-intensive Math backend instead of using FlashAttention. Second, we are migrating to Grouped Query Attention (GQA), and we need a transparent reference implementation of SDPA in NumPy to cross-check tensor outputs and verify correctness independently of the hardware backend.

Your task is to complete three components:

1. **Reference SDPA:** Implement `numpy_sdpa` in `sdpa/reference.py` using NumPy. It must handle GQA by correctly repeating Key/Value heads to match Query heads, apply causal masking by adding `-inf` to future tokens, and apply an optional custom boolean mask (`True` means keep, `False` means mask).
2. **Backend Dispatcher:** Implement `predict_backend` and `repair_config_for_flash` in `sdpa/dispatch.py` and `sdpa/repair.py`. We need to predict which backend PyTorch will choose based on our simplified rules, and we need an automatic repair function that takes a rejected configuration dictionary and mutates it minimally to force FlashAttention.
3. **Safety Net:** Write a regression test in `tests/test_regression.py` that verifies your causal masking logic. The harness will test your test suite by injecting a fault that completely ignores the `is_causal` flag. Your test must assert an invariant that catches this failure.

### Dispatch Rules
- **FlashAttention:** `dtype` in `("float16", "bfloat16")`, `head_dim` in `(16, 32, 64, 128, 256)`, and `has_custom_mask` is `False`.
- **Memory Efficient:** `dtype` in `("float16", "bfloat16", "float32")`, `head_dim` is a multiple of 8 (up to 128), and `has_custom_mask` is `False`.
- **Math:** Fallback if neither above applies.

### Config Repair Rules
Given a config dictionary, mutate it to guarantee FlashAttention is chosen:
- `head_dim`: Round UP to the nearest valid Flash dimension. If greater than 256, clamp it to 256.
- `dtype`: If not Flash-compatible, change it to `"float16"`.
- `has_custom_mask`: If `True`, set it to `False`, and simultaneously set `is_causal` to `True` to preserve some masking capability.
