## Context

In transformer language models, the key‑value (KV) cache stores intermediate representations for each token generated so far. For a model with \(L\) layers and \(H\) attention heads per layer, each head has dimension \(d\). The cache holds two tensors per head: keys and values, both of shape \((C,d)\), where \(C\) is the current context length (number of tokens decoded). If the cache uses a NumPy dtype with item size \(\mathrm{size}(t)\) bytes, then the total memory footprint in bytes is

$$
B = 2\,L\,H\,C\,d\,\mathrm{size}(t).
$$

The factor \(2\) accounts for keys and values. The problem asks to implement a function that returns this closed‑form byte count.

## Task

Implement `kv_cache_bytes(layers, heads, head_dim, seq_len, dtype)`:

```python
def kv_cache_bytes(layers: int,
                   heads: int,
                   head_dim: int,
                   seq_len: int,
                   dtype: str) -> int:
    ...
```

The function should return the exact number of bytes required to store the KV cache for a single‑token batch. Use NumPy’s `dtype` machinery to obtain the item size; do **not** hard‑code sizes such as 4 or 8.

## Example

```python
>>> kv_cache_bytes(12, 16, 128, 2048, "float16")
8388608
```

Explanation:  
\(2 \times 12 \times 16 \times 2048 \times 128 \times 2 = 8\,388\,608\) bytes.

## What the gate checks

The grader computes a reference byte count using NumPy’s `dtype.itemsize`. The candidate must return an integer equal to that value. The metric `size_ratio` is used; it should be exactly \(1.0\). Any deviation causes the gate to fail.
