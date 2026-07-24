## Context

In transformer models the attention mechanism requires three linear projections of the input token embeddings: queries $Q$, keys $K$ and values $V$. For efficiency many implementations fuse these three projections into a single matrix multiplication followed by a split. Additionally, Rotary Positional Embedding (RoPE) injects absolute position information by rotating pairs of embedding dimensions in the key and value tensors:

$$
\\begin{aligned}
k'_{i,2j} &= k_{i,2j}\\cos(\\theta_j) - k_{i,2j+1}\\sin(\\theta_j),\\\\
k'_{i,2j+1} &= k_{i,2j}\\sin(\\theta_j) + k_{i,2j+1}\\cos(\\theta_j),
\\end{aligned}
$$

where $\\theta_j = \\omega_j\,p_i$ with $p_i$ the position of token $i$ and $\\omega_j$ a frequency schedule.

The KV cache stores all past keys and values so that subsequent decoding steps can reuse them. A common pattern is to write the rotated $K$ and $V$ into the cache at the current sequence position in one pass, avoiding an extra copy.

## Task

Implement `fused_qkv_rope_kv_cache_write`:

```python
def fused_qkv_rope_kv_cache_write(
    x: np.ndarray,
    weight_q: np.ndarray,
    weight_k: np.ndarray,
    weight_v: np.ndarray,
    rope_freqs: np.ndarray,
    kv_cache_k: np.ndarray,
    kv_cache_v: np.ndarray,
    cache_pos: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `x` – input token embeddings of shape `(batch, seq_len, d_model)`.
* `weight_q`, `weight_k`, `weight_v` – projection matrices of shape `(d_model, d_model)`.
* `rope_freqs` – 1‑D array of length `d_model//2` containing the base frequencies $\\omega_j$.
* `kv_cache_k`, `kv_cache_v` – mutable arrays that hold all past keys and values. They have shape `(batch, max_seq_len, d_model)` and will be updated in place at indices `[ :, cache_pos:cache_pos+seq_len, : ]`.
* `cache_pos` – integer offset into the KV cache where the new tokens should be written.

The function must:

1. Compute $Q = XW_Q$, $K = XW_K$, $V = XW_V$ using matrix multiplication.
2. Apply RoPE to $K$ and $V$ as described above, producing `k_rot` and `v_rot`.
3. Write `k_rot` into `kv_cache_k` and `v_rot` into `kv_cache_v` at the given position.
4. Return the tuple `(q, k_rot, v_rot)`.

All operations must use NumPy vectorized code; no explicit Python loops over tokens or heads are allowed.

## Example

```python
import numpy as np

batch, seq_len, d = 2, 3, 6
x = np.random.randn(batch, seq_len, d)
wq = np.random.randn(d, d)
wk = np.random.randn(d, d)
wv = np.random.randn(d, d)

# frequencies for RoPE: ω_j = 1 / (10000^(j/d))
rope_freqs = 1.0 / (10000 ** (np.arange(d//2) / d))

kv_cache_k = np.zeros((batch, 10, d), dtype=np.float64)
kv_cache_v = np.zeros((batch, 10, d), dtype=np.float64)

q, k_rot, v_rot = fused_qkv_rope_kv_cache_write(
    x, wq, wk, wv,
    rope_freqs,
    kv_cache_k, kv_cache_v,
    cache_pos=4
)
```

After the call `kv_cache_k[:, 4:7]` and `kv_cache_v[:, 4:7]` contain the rotated keys and values.

## What the gate checks

The grader computes a reference implementation using NumPy. It then compares:

* The returned $Q$, $K_{\text{rot}}$ and $V_{\text{rot}}$ to the reference with `max_abs_err`.
* That the provided cache arrays have been updated exactly as expected.

The maximum absolute difference must be at most $10^{-5}$.
