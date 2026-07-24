## Context

In transformer‑based language models each layer stores a *key* and a *value* tensor for every token in the batch.  
For a single layer the memory required to hold these tensors is

$$
\text{layer}_\text{kv} = 2 \times B \times H \times S \times D \times b,
$$

where  

- $B$ – batch size,  
- $H$ – number of attention heads,  
- $S$ – sequence length,  
- $D$ – head dimension,  
- $b$ – bytes per element (the `dtype.itemsize`).

Two common strategies for managing this memory across the $L$ layers are:

1. **Offloaded double buffer** – only two layers’ KV tensors reside in device memory at any time; the rest are off‑loaded to host or disk.  
   The peak resident bytes are therefore

   $$
   \text{peak}_\text{off} = 2 \times \text{layer}_\text{kv}.
   $$

2. **On‑device full cache** – all layers keep their KV tensors in device memory simultaneously, giving

   $$
   \text{peak}_\text{full} = L \times \text{layer}_\text{kv}.
   $$

The task is to compute these two peak values for arbitrary model and batch parameters.

## Task

Implement the function `kv_peak_bytes`:

```python
def kv_peak_bytes(num_layers: int,
                  batch_size: int,
                  num_heads: int,
                  seq_len: int,
                  head_dim: int,
                  dtype: np.dtype) -> tuple[float, float]:
    """
    Return (peak_offloaded_bytes, peak_full_cache_bytes).
    """
```

The function should use only NumPy and basic arithmetic; no loops are required.

## Example

```python
import numpy as np
from kv_peak import kv_peak_bytes  # assume the module is named kv_peak.py

# Model with 12 layers, batch of 8, 12 heads, sequence length 128,
# head dimension 64, and float32 weights.
peak_off, peak_full = kv_peak_bytes(
    num_layers=12,
    batch_size=8,
    num_heads=12,
    seq_len=128,
    head_dim=64,
    dtype=np.float32
)

print(peak_off)   # 2 * 8 * 12 * 128 * 64 * 4 = 7864320 bytes
print(peak_full)  # 12 * that = 94371840 bytes
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares the returned tuple to the expected values. The comparison is exact (`==`), so any mismatch, including type or rounding errors, will fail the gate.
