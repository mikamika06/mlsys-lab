## Context

In transformer models the attention mechanism stores two tensors per layer for each token in the context: a key matrix $K$ and a value matrix $V$.  
For a model with hidden dimension $d_{\text{model}}$, number of heads $h$, and context length $L$, the shape of these matrices depends on the attention variant.

* **Multi‑Head Attention (MHA)** – each head has its own key/value projection.  
  The per‑head dimension is $d_h = d_{\text{model}}/h$.  
  For one layer we store
  \[
    K, V \in \mathbb{R}^{L\times h\times d_h}.
  \]
  The total byte size for a single layer is therefore
  \[
    S_{\text{MHA}}
      = 2\,L\,h\,d_h\,\mathrm{bytes}
      = 2\,L\,d_{\text{model}}\,\mathrm{bytes},
  \]
  where the factor $2$ accounts for both $K$ and $V$.

* **Grouped‑Query Attention (GQA)** – keys/values are shared across heads within a group.  
  If we split the $h$ heads into $g$ groups, each group contains $h/g$ heads that share the same key/value matrices.  
  The per‑group shape is
  \[
    K_g, V_g \in \mathbb{R}^{L\times g\times d_h},
  \]
  giving a layer size of
  \[
    S_{\text{GQA}}
      = 2\,L\,g\,d_h\,\mathrm{bytes}
      = 2\,L\,d_{\text{model}}\frac{g}{h}\,\mathrm{bytes}.
  \]

* **Multi‑Query Attention (MQA)** – all heads share a single key/value pair.  
  The shape per layer is
  \[
    K_q, V_q \in \mathbb{R}^{L\times d_h},
  \]
  so the byte size becomes
  \[
    S_{\text{MQA}}
      = 2\,L\,d_h\,\mathrm{bytes}
      = 2\,L\,\frac{d_{\text{model}}}{h}\,\mathrm{bytes}.
  \]

The byte count for a full model with $N$ layers is simply the per‑layer size multiplied by $N$.  
All sizes are computed using NumPy arrays of the specified data type; the number of bytes per element is given by `np.dtype(dtype).itemsize`.

## Task

Implement the function

```python
def kv_cache_bytes(
    layers: int,
    ctx_len: int,
    d_model: int,
    num_heads: int,
    groups: int = 1,
    dtype=np.float32,
) -> dict:
```

The function must return a dictionary with keys `"mha"`, `"gqa"` and `"mqa"`.  
Each value is the total number of bytes required to store the key/value cache for that attention variant across all layers.  
Use NumPy arrays to compute the sizes; do **not** hard‑code any numeric constants.

## Example

```python
import numpy as np

bytes_dict = kv_cache_bytes(
    layers=4,
    ctx_len=2048,
    d_model=4096,
    num_heads=32,
    groups=8,
    dtype=np.float16,
)

print(bytes_dict)
# {'mha': 268435456, 'gqa': 67108864, 'mqa': 8388608}
```

## What the gate checks

The grader computes the reference sizes using NumPy and compares two ratios:

* `gqa_mha_ratio = size_gqa / size_mha`
* `mqa_mha_ratio = size_mqa / size_mha`

Both ratios must match the oracle within a relative tolerance of $10^{-9}$.  The gate reports success only if both metrics equal `1.0` (i.e., the computed ratios are correct to the required precision).
