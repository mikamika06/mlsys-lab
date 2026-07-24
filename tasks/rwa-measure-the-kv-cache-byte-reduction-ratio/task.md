## Context

In a transformer decoder, the *key/value* (KV) cache stores the hidden states of all previously generated tokens so that each new token can attend to them without recomputing the attention scores.  
For one additional token the cache grows by

$$
2 \times H_{\text{kv}} \times D_{\text{head}} \times B,
$$

where $H_{\text{kv}}$ is the number of KV heads, $D_{\text{head}}$ the dimensionality of each head and $B$ the byte size of the underlying data type (e.g. $4$ for `float32`).  
A full multi‑head attention (MHA) implementation would use $H_{\text{q}}$ query heads, so its incremental cost is

$$
2 \times H_{\text{q}} \times D_{\text{head}} \times B.
$$

The *byte reduction ratio* of a KV cache relative to a full MHA is therefore

$$
\frac{\text{bytes per token for KV cache}}
     {\text{bytes per token for MHA}}
= \frac{H_{\text{kv}}}{H_{\text{q}}},
$$

independent of $D_{\text{head}}$ and $B$.  In GQA (Grouped‑Query Attention) the number of KV heads is deliberately reduced, yielding a predictable savings factor.

## Task

Implement `kv_cache_stats` that returns three values:

```python
def kv_cache_stats(
    n_kv_heads: int,
    n_q_heads: int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> tuple[int, int, float]:
```

* `bytes_per_token_kv`: the incremental number of bytes added to the KV cache for one new token.  
* `bytes_per_token_mha`: the incremental number of bytes that a full MHA would add for one new token.  
* `ratio`: the byte reduction ratio defined above.

All arithmetic should use integer types where appropriate; the ratio must be returned as a Python float.  The function should work for any positive integers.

## Example

```python
>>> kv_cache_stats(8, 16, 64)
(4096, 8192, 0.5)

# bytes_per_token_kv = 2 * 8 * 64 * 4 = 4096
# bytes_per_token_mha = 2 * 16 * 64 * 4 = 8192
# ratio = 4096 / 8192 = 0.5
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares each of the three returned values against it:

* The two integer byte counts must match exactly.  
* The floating‑point ratio must be within $10^{-12}$ relative error of the reference.

If all checks pass, the solution receives an `exact_match` score of $1.0$; otherwise $0.0$.  A correct implementation will therefore satisfy the gate with a perfect score.
