## Context

In a multi‑head attention block each head operates on a contiguous slice of the query, key and value tensors.  
If the total dimensionality of the heads is $H \times d_k$ for queries (and keys/values) and $H \times d_v$ for the output projection, then head $h$ ($0\le h < H$) owns columns
$$
q_{\text{cols}} = [\,h\,d_k,\;(h+1)\,d_k\,),
$$
similarly for $k$ and $v$.  
The output projection matrix $W_o \in \mathbb R^{(H d_v)\times D}$ takes as input the concatenated values; head $h$ therefore owns rows
$$
W_{o,\text{rows}} = [\,h\,d_v,\;(h+1)\,d_v\,).
$$

The coupling map is the set of these index ranges for a given head.  It tells which parts of each tensor must be removed together when pruning that head.

## Task

Implement `classify_coupling_map`:

```python
def classify_coupling_map(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    o_proj_shape: tuple[int, ...],
    head_index: int,
) -> dict[str, tuple[int, int]]:
    ...
```

The function receives the shapes of the four tensors involved in a standard multi‑head attention block and an integer `head_index`.  
It must return a dictionary mapping each tensor name to a two‑tuple `(start, end)` describing the slice indices that belong to the requested head.  The keys are `"q"`, `"k"`, `"v"` and `"o_proj_input"`.

All slices should be half‑open intervals `[start, end)`.  
The function must not perform any array operations; it only needs to compute integer indices from the shapes.

## Example

```python
import numpy as np

# 4 heads, each of size 16 in Q/K/V and 16 in O_proj rows
q_shape = (2, 10, 64)          # batch=2, seq_len=10, H*d_k=64
k_shape = q_shape
v_shape = q_shape
o_proj_shape = (64, 128)       # H*d_v=64, embed_dim=128

head_index = 1                # second head

mapping = classify_coupling_map(q_shape, k_shape, v_shape, o_proj_shape, head_index)
print(mapping)
# {'q': (16, 32), 'k': (16, 32), 'v': (16, 32), 'o_proj_input': (16, 32)}
```

## What the gate checks

The grader computes a reference coupling map using the same arithmetic as in the description and compares it to your output with an exact match.  The metric `exact_match` must equal `1.0`.  No other metrics are evaluated.
