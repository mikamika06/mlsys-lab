## Context

An inference server's KV-cache for one request is two tensors, $K$ and $V$,
each of shape $(\text{layers}, \text{heads}, \text{seq\_len}, \text{head\_dim})$
and some fixed dtype. To checkpoint a request (e.g. to move it between GPU
and host memory, or across machines for disaggregated prefill/decode), the
cache has to be packed into a flat byte buffer that carries enough metadata
to reconstruct the exact tensors later — this is the same problem every
tensor-serialization format (safetensors, `torch.save`, raw checkpoint
shards) solves with a small fixed header in front of the raw bytes.

## Task

Implement a pair of functions that serialize a KV-cache to a single `bytes`
blob and back, **byte-for-byte exact**:

```python
def pack_kv_cache(K: np.ndarray, V: np.ndarray) -> bytes: ...
def unpack_kv_cache(blob: bytes) -> tuple[np.ndarray, np.ndarray]: ...
```

* `K`, `V` — NumPy arrays, both of shape
  $(\text{layers}, \text{heads}, \text{seq\_len}, \text{head\_dim})$ and the
  **same** dtype, one of `float32`, `float16`, `int8`, `float64`.

**Wire format** (all integers little-endian):

| bytes | field | type |
|---|---|---|
| 1 | `dtype_code` | `uint8` — `0`=float32, `1`=float16, `2`=int8, `3`=float64 |
| 4 | `num_layers` | `uint32` |
| 4 | `num_heads` | `uint32` |
| 4 | `seq_len` | `uint32` |
| 4 | `head_dim` | `uint32` |
| $\lvert K\rvert$ | `K` data | raw C-contiguous bytes of `K` |
| $\lvert V\rvert$ | `V` data | raw C-contiguous bytes of `V`, same length as `K`'s |

The header is exactly 17 bytes:
`struct.pack("<BIIII", dtype_code, num_layers, num_heads, seq_len, head_dim)`.
`pack_kv_cache` must produce exactly this layout; `unpack_kv_cache` must
correctly invert it, including cases where the input blob it receives was
**not** produced by your own `pack_kv_cache` (e.g. a reference blob built
independently from the same spec) — so don't smuggle extra state through a
side channel, the blob's bytes are the only source of truth.

## Example

```python
import numpy as np
K = np.random.default_rng(0).standard_normal((2, 4, 8, 16)).astype(np.float32)
V = np.random.default_rng(1).standard_normal((2, 4, 8, 16)).astype(np.float32)

blob = pack_kv_cache(K, V)
K2, V2 = unpack_kv_cache(blob)
assert K2.shape == K.shape and K2.dtype == K.dtype
assert np.array_equal(K, K2) and np.array_equal(V, V2)
```

## What the gate checks

For several `(layers, heads, seq_len, head_dim, dtype)` cases with seeded
random `K`/`V`:

* Your `pack_kv_cache(K, V)` output is compared **byte-for-byte** against an
  independently built reference blob using the exact layout above (same
  length, `byte_exact_fraction == 1.0`).
* Your `unpack_kv_cache` is called on the **reference** blob (not your own
  packed output), and the returned `K`, `V` must match the original arrays'
  shape, dtype, and bytes exactly.

A single gate, **byte_exact_fraction**, is `1.0` only if every case above
passes; any mismatch (wrong header layout, wrong byte order, wrong dtype
mapping, wrong reshape) fails it to `0.0`.
