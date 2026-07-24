## Context

In a multi‑head attention module, each query head typically attends to the same key/value (KV) tensors that are shared across several heads.  
When KV is *packed* for efficiency, we store only one copy per group of heads.  
Let $H$ be the number of packed KV heads and $n_{\text{rep}}$ the replication factor: each packed head serves $n_{\text{rep}}$ query heads.  
The full per‑query‑head view would therefore have shape $(H \times n_{\text{rep}}, L, D)$ where $L$ is the sequence length and $D$ the dimensionality of the KV vectors.

Reconstructing this view means expanding the packed tensor along its first dimension so that each query head receives the correct KV slice.  
Mathematically, for a packed tensor $\mathbf{K}\in\mathbb{R}^{H\times L\times D}$ we want

$$
\hat{\mathbf{K}}_{h,\ell,d} = \mathbf{K}_{\,\lfloor h / n_{\text{rep}}\rfloor ,\,\ell,\,d},
\qquad
h=0,\dots,Hn_{\text{rep}}-1.
$$

## Task

Implement the function `unpack_gqa`:

```python
import numpy as np

def unpack_gqa(kv_packed: np.ndarray, n_rep: int) -> np.ndarray:
    ...
```

The input `kv_packed` is a 3‑D NumPy array of shape `(H, L, D)` and the integer `n_rep > 0`.  
Return a new array of shape `(H * n_rep, L, D)` that contains each packed head repeated exactly `n_rep` times along the first axis.

The implementation must use only NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np

kv_packed = np.array([[[1], [2]], [[3], [4]]])  # shape (2, 2, 1)
n_rep = 3
expanded = unpack_gqa(kv_packed, n_rep)

print(expanded.shape)          # (6, 2, 1)
print(expanded[0])             # [[1], [2]]
print(expanded[3])             # [[3], [4]]
```

## What the gate checks

The grader computes a reference expansion using NumPy’s `repeat` and compares element‑wise with your output.  
Your solution must match exactly for all provided test cases; otherwise the gate fails.
