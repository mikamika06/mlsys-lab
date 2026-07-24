## Context

In transformer models, the key‑value (KV) cache stores intermediate tensors that are reused during decoding. For a model with \(L\) layers and \(H\) attention heads, each head holds a key matrix \(K \in \mathbb{R}^{B\times T_{\text{max}}\times d_k}\) and a value matrix \(V \in \mathbb{R}^{B\times T_{\text{max}}\times d_v}\), where \(B\) is the batch size, \(T_{\text{max}}\) the maximum sequence length, and \(d_k=d_v\) the head dimensionality. The total number of floating‑point elements in the cache is therefore

$$
N = L \cdot H \cdot 2 \cdot B \cdot T_{\text{max}} \cdot d_k .
$$

If each element occupies \(s=\mathrm{sizeof}(\text{dtype})\) bytes, the total memory footprint in bytes is

$$
M = N \cdot s .
$$

The task is to implement a function that returns this exact byte count for arbitrary integer arguments and a NumPy dtype.

## Task

Implement `kv_cache_bytes(layers: int, heads: int, d_kv: int, seq_len: int, dtype: np.dtype, batch: int) -> int`:

```python
def kv_cache_bytes(layers, heads, d_kv, seq_len, dtype, batch):
    ...
```

The function must compute the total number of bytes required to store all key and value tensors for a full KV cache given the model configuration. It should return an integer.

## Example

```python
import numpy as np
bytes_needed = kv_cache_bytes(
    layers=2,
    heads=8,
    d_kv=64,
    seq_len=128,
    dtype=np.float32,
    batch=4
)
print(bytes_needed)  # 2 * 8 * 2 * 4 * 128 * 64 * 4 = 4194304
```

## What the gate checks

The grader computes an analytic reference using NumPy and compares it to your result with a simple ratio. The ratio must be exactly 1.0 (within a tolerance of \(10^{-12}\)). Any deviation causes the task to fail.
