## Context

During autoregressive text generation a transformer language model produces one
token per step. Each new token must attend to the **entire** history stored in
the KV cache — the key and value tensors saved from every previous position.

For a model with $L$ layers, $H$ attention heads of dimension $d_h$, context
length $T$, and element size $b$ bytes, the KV cache for one layer holds:

$$\text{single-layer bytes} = 2 \cdot H \cdot d_h \cdot T \cdot b$$

The factor of 2 accounts for both K (key) and V (value). Across all layers
the full cache is

$$\text{KV cache bytes} = L \cdot 2 \cdot H \cdot d_h \cdot T \cdot b$$

In a decode step the attention kernel reads every element of the KV cache to
compute attention scores and the weighted sum, so the bytes moved from memory
equal the full cache size. This is what makes autoregressive decoding
**memory-bandwidth bound**: arithmetic intensity drops as context length grows
because the model performs $O(T)$ work per output token while reading
$O(T)$ data, yet the ratio stays constant regardless of how many FLOPs the
matmuls inside the attention block perform.

Understanding this quantity matters for profiling and for deciding when
techniques like KV-cache quantization, multi-query attention, or paged caches
are worth the engineering effort.

## Task

Implement `kv_bytes_per_decode`:

```python
def kv_bytes_per_decode(
    n_layers: int,
    n_heads: int,
    d_head: int,
    ctx_len: int,
    dtype_bytes: int,
) -> int:
```

Return the total number of bytes read from the KV cache during one
autoregressive decode step.

Parameters:

| Name | Symbol | Meaning |
|---|---|---|
| `n_layers` | $L$ | number of transformer layers |
| `n_heads` | $H$ | attention heads per layer |
| `d_head` | $d_h$ | dimension per head |
| `ctx_len` | $T$ | context length (number of cached tokens) |
| `dtype_bytes` | $b$ | bytes per cache element (2 for fp16, 4 for fp32, …) |

Return a non-negative integer.

## Example

```python
# GPT-2 small: 12 layers, 12 heads, d_head=64, 1024 ctx, fp16 (b=2)
kv_bytes_per_decode(12, 12, 64, 1024, 2)
# => 37,748,736

# LLaMA-7B: 32 layers, 32 heads, d_head=128, 2048 ctx, fp16
kv_bytes_per_decode(32, 32, 128, 2048, 2)
# => 1,073,741,824
```

Derivation of the first example:

$$12 \times 2 \times 12 \times 64 \times 1024 \times 2 = 37{,}748{,}736$$

## What the gate checks

One gate: `bytes_formula_correct`. Five test cases span small to large model
configs with dtype sizes of 1, 2, 4, and 8 bytes. The reference answer is
computed independently using NumPy's `dtype.itemsize` oracle to confirm the
byte width, then evaluating $L \times 2 \times H \times d_h \times T \times
b$. Your answer must match the integer reference exactly on every case.
