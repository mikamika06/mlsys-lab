## Context

Large language model inference often stores key-value (KV) cache data in high-bandwidth memory (HBM). If every transformer layer keeps its KV cache resident, the memory requirement grows with the number of layers.

For a model with $n_{\text{layers}}$ layers and a KV cache size of $b$ bytes per layer, keeping all layers resident requires

$$
B_{\text{full}} = n_{\text{layers}} \cdot b .
$$

Layer-wise offload scheduling keeps only $k$ active layers in HBM and moves other layers to another memory tier when they are not needed. The peak resident memory becomes

$$
B_{\text{resident}} = k \cdot b .
$$

The saved-memory ratio can be expressed as the amount of full-model HBM that remains resident:

$$
r = \frac{B_{\text{resident}}}{B_{\text{full}}}
= \frac{k \cdot b}{n_{\text{layers}} \cdot b}.
$$

A smaller ratio means less HBM is occupied by the resident KV cache.

## Task

Implement `measure_hbm_saved(n_layers, active_layers, per_layer_kv_bytes)`:

```python
def measure_hbm_saved(
    n_layers: int,
    active_layers: int,
    per_layer_kv_bytes: int,
) -> dict:
    ...
```

Return a dictionary with exactly these keys:

- `"peak_resident_bytes"`: the peak HBM bytes occupied by resident KV cache after scheduling.
- `"resident_ratio"`: the fraction of full KV-cache HBM that remains resident.

The function should compute the values using the layer-wise offload formula. Assume $1 \leq k \leq n_{\text{layers}}$ and all byte values are non-negative integers.

## Example

```python
result = measure_hbm_saved(
    n_layers=32,
    active_layers=4,
    per_layer_kv_bytes=1024 * 1024,
)

# {
#   "peak_resident_bytes": 4194304,
#   "resident_ratio": 0.125
# }
```

## What the gate checks

The gate computes the expected resident memory using an independent oracle implementation of the scheduling formula.

The returned `"peak_resident_bytes"` and `"resident_ratio"` values must exactly match the oracle for several layer counts and KV sizes. The `size_ratio` score is $1.0$ only when the implementation reports the correct resident memory reduction behavior.
