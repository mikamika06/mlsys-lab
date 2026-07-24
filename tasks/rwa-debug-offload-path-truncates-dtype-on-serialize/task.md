## Context

Key-value (KV) caches used during model inference are often moved between memory tiers. An offload path serializes a cache tensor, stores the bytes, and reloads it later.

A cache tensor $K$ with values in floating point format should preserve its numerical representation across this round trip:

$$
K_{\mathrm{loaded}} = K_{\mathrm{original}} .
$$

A lossy serializer can silently cast values to a narrower dtype. For example, converting a `float32` tensor to `float16` changes the stored values because `float16` has fewer mantissa bits. The error for a tensor is measured as

$$
\max_{i} |K_{\mathrm{loaded},i} - K_{\mathrm{original},i}| .
$$

Even small errors can accumulate when the cache is reused by attention layers.

## Task

Debug the KV offload serialization path by implementing `serialize_kv_roundtrip`:

```python
def serialize_kv_roundtrip(kv: np.ndarray) -> np.ndarray:
    ...
```

The function receives a NumPy array representing cached key/value data and returns the array after a serialize and deserialize round trip.

The implementation must preserve the original values. The returned array should contain the same numerical values as the input array and should not intentionally narrow the dtype during serialization.

## Example

```python
import numpy as np

kv = np.array([0.1234567, -3.1415926], dtype=np.float32)
restored = serialize_kv_roundtrip(kv)

# restored values match kv within floating point precision
```

## What the gate checks

The gate builds a NumPy reference by treating the original KV cache as the lossless serialization target in higher precision. It measures

$$
\max_i |x_i - \hat{x}_i|
$$

between the oracle values and the candidate round trip result.

A lossy implementation that converts the cache to `float16` during serialization has an error above the required threshold. The fixed implementation must achieve `max_abs_err \le 10^{-6}`.
