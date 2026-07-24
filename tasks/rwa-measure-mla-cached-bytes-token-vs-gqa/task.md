## Context

In transformer models the key‑value (KV) cache is often stored per token.  
For a standard *GQA* (global query attention) implementation each head stores a full KV vector of dimension `head_dim`.  
If there are `n_kv_heads` heads, the number of elements cached for one token is

$$
2 \times n_{\text{kv\_heads}} \times \text{head\_dim},
$$

the factor 2 accounting for key and value.  

The *MLA* (memory‑efficient linear attention) variant replaces this full KV vector with a low‑rank latent of size `kv_lora_rank`.  
Thus the number of elements cached per token is simply

$$
\text{kv\_lora\_rank}.
$$

When these numbers are multiplied by the byte width of the chosen data type (e.g. 2 bytes for `float16`, 4 bytes for `float32`) we obtain the cache size in bytes.

## Task

Implement a function that, given the number of KV heads, head dimension, low‑rank latent size and a NumPy dtype, returns the cached bytes per token for both MLA and GQA:

```python
def cached_bytes_per_token(
    n_kv_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    dtype: np.dtype
) -> tuple[int, int]:
    """
    Return (mla_bytes_per_token, gqa_bytes_per_token).
    """
```

The function must use `np.dtype(dtype).itemsize` to obtain the byte width of the data type.

## Example

```python
import numpy as np

bytes_mla, bytes_gqa = cached_bytes_per_token(
    n_kv_heads=8,
    head_dim=64,
    kv_lora_rank=32,
    dtype=np.float16
)
print(bytes_mla)  # 64   (32 * 2)
print(bytes_gqa)  # 2048 (2 * 8 * 64 * 2)
```

## What the gate checks

The grader verifies two things:

1. **Exact match** – the tuple returned by your function must equal the oracle’s tuple of integers.
2. **Size ratio error** – it computes the cached‑bytes ratio `mla_bytes / gqa_bytes` and compares it to the oracle’s value; the absolute difference must be ≤ $10^{-12}$.

Both checks are performed on a fixed test case (`n_kv_heads=8`, `head_dim=64`, `kv_lora_rank=32`, `dtype=np.float16`).  
A correct implementation passes both gates, while any deviation (e.g. missing the factor 2 for GQA) fails at least one gate.
