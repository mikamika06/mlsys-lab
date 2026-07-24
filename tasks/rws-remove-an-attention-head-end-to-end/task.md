## Context

Multi-head attention splits a model dimension into independent attention heads. If the
model dimension is $d$ and there are $H$ heads, each head has width

$$d_h = \frac{d}{H}.$$

The query, key, and value projections contain one output slice per head. For head $h$,
the slice

$$[h d_h : (h+1)d_h]$$

must be removed from the output dimensions of the $Q$, $K$, and $V$ projections. The
output projection consumes the concatenated head outputs, so the same slice must be
removed from the input rows of $W_o$.

After removing a head, the remaining attention block has dimension

$$d' = d - d_h.$$

For input $X$, the projections are

$$Q = XW_q,\qquad K = XW_k,\qquad V = XW_v.$$

Each remaining head computes scaled dot-product attention:

$$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.$$

The head outputs are concatenated and passed through the pruned output projection.

## Task

Implement `remove_attention_head`:

```python
def remove_attention_head(
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    x: np.ndarray,
    head: int,
    num_heads: int,
):
    ...
```

The function receives square projection matrices of shape $(d,d)$, an input matrix
$x$ of shape $(n,d)$, the head index to remove, and the original number of heads.

Return:

```python
(Wq_pruned, Wk_pruned, Wv_pruned, Wo_pruned, output)
```

where the first four values are the sliced matrices and `output` is the attention block
output after removing the selected head.

Use NumPy operations only. The attention computation must use the remaining heads in
their original order.

## Example

```python
import numpy as np

d = 8
H = 4
x = np.zeros((2, d))

Wq2, Wk2, Wv2, Wo2, y = remove_attention_head(
    np.eye(d), np.eye(d), np.eye(d), np.eye(d), x, 1, H
)

# The removed head has width 2, so the pruned projections use dimension 6.
# y has shape (2, 8).
```

## What the gate checks

The gate builds an independent NumPy oracle that slices the coupled projection
matrices, recomputes the remaining multi-head attention, and compares the returned
block output. The maximum absolute error must satisfy

$$\max_i |y_i - y_i^{\mathrm{oracle}}| \le 10^{-6}.$$

Returning only sliced weights, dropping the wrong projection axis, or leaving the
output projection unchanged will produce a numerical mismatch.
