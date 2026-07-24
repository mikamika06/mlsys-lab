## Context

In transformer‑based language models, the key–value (KV) cache stores intermediate activations that are reused across decoding steps.  
For a single request the amount of data that must be transferred to the GPU is proportional to the size of this KV cache.

If a model has

* $L$ layers,
* $H_k$ KV heads per layer,
* head dimension $D_h$,
* sequence length $S$ (the number of tokens in the request),
* and each scalar occupies $B$ bytes (e.g. 4 for `float32`),

then the total number of bytes that must be moved is

$$
\text{bytes} \;=\;
2 \times L \times H_k \times D_h \times S \times B .
$$

The factor $2$ accounts for both key and value tensors.

## Task

Implement a function `kv_cache_transfer_bytes` that receives the five integer arguments described above and returns the exact number of bytes that must be transferred for the request.

```python
def kv_cache_transfer_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    seq_len: int
) -> int:
    ...
```

The function should perform only integer arithmetic and return an `int`.

## Example

```python
>>> kv_cache_transfer_bytes(12, 16, 64, 4, 128)
1966080
```

Explanation: $2 \times 12 \times 16 \times 64 \times 128 \times 4 = 1\,966\,080$.

## What the gate checks

The grader computes the reference value using the same closed‑form formula and compares it to your output.  
Your implementation must match exactly; any mismatch or exception causes the gate to fail.
