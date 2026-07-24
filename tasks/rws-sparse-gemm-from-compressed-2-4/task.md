## Context

NVIDIA's Sparse Tensor Cores don't store a 2:4-pruned weight matrix
(every group of 4 consecutive columns has exactly 2 nonzeros) as a
dense matrix with zeros, nor as generic (row, col, value) triples.
They use a **fixed, dense compressed layout**: per group of 4 columns,
store the **2 nonzero values**, plus a **2-bit index per value** (4
possible positions, 2 bits each) recording which of the 4 columns in
the group it came from. This halves both the value storage *and* the
metadata is tiny (2 bits per kept value instead of a full column index),
and — critically — a real sparse-tensor-core kernel computes the matmul
**directly from this compressed layout**, without ever materializing
the dense zero-padded matrix.

### Layout

For a weight matrix $W \in \mathbb{R}^{d_{out}\times d_{in}}$ ($d_{in}$
a multiple of 4) that is 2:4-sparse (exactly 2 nonzeros per row per
group of 4 consecutive columns), the compressed representation is:

* `values` $\in \mathbb{R}^{d_{out}\times d_{in}/2}$ — for row $r$ and
  group $g = 0,\dots,d_{in}/4-1$, slots `values[r, 2g]` and
  `values[r, 2g+1]` hold that group's two nonzero weights, in ascending
  order of their original column position within the group.
* `idx` — same shape as `values`, integers in $\{0,1,2,3\}$: `idx[r, k]`
  is the position **within its group of 4** that `values[r, k]` came
  from. The nonzero's true column is $4g + \mathrm{idx}[r, 2g{+}s]$ for
  slot $s\in\{0,1\}$.

Given this layout and an activation matrix $X \in
\mathbb{R}^{d_{in}\times n}$, the layer computes $Y = WX \in
\mathbb{R}^{d_{out}\times n}$ — using only `values`, `idx`, and `X` (the
dense $W$, with its zeros, is never stored).

## Task

Implement:

```python
def compressed_matmul(values: np.ndarray, idx: np.ndarray, X: np.ndarray) -> np.ndarray:
    ...
```

* `values` — `(d_out, d_in//2)` nonzero weights (2 per group of 4).
* `idx` — `(d_out, d_in//2)` integers in `{0,1,2,3}`, the within-group
  position of each value (same shape and slot order as `values`).
* `X` — `(d_in, n)` activations.

Return `Y = W @ X`, where `W` is the dense matrix implied by the
compressed layout above.

## Example

```python
import numpy as np
# one row, one group of 4: kept columns 1 and 3, with values 5.0 and -2.0
values = np.array([[5.0, -2.0]])
idx = np.array([[1, 3]])
X = np.eye(4)   # so W @ X reveals W directly
Y = compressed_matmul(values, idx, X)
# Y == [[0, 5.0, 0, -2.0]]  (the dense row W reconstructed at columns 1 and 3)
```

## What the gate checks

* **max_abs_err** — your `Y` must match a NumPy oracle that reconstructs
  the same dense `W` from `(values, idx)` and computes `W @ X`, to
  within $10^{-6}$ absolute error, over several random 2:4-sparse
  weight matrices and activation shapes (fixed seed).
