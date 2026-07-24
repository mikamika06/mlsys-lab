## Context

NVIDIA's Ampere+ sparse tensor cores accelerate matmuls under **2:4
structured sparsity**: every group of 4 consecutive columns in a weight
row has exactly 2 zeros. The hardware never touches the zeros — it stores
only the 2 surviving **values** per group plus a 2-bit **index** (`0..3`)
recording each value's original position, and reconstructs the operand
from that compressed form immediately before the matmul.

This compressed execution path is a *different sequence of operations*
from a plain dense matmul on the same (already-sparse) matrix — but if the
hardware/kernel is implemented correctly, it must produce **numerically
identical** output. There is no approximation here: 2:4 compression of an
already-2:4 matrix followed by exact reconstruction is lossless. Any gap
between the two paths' outputs is a real correctness bug (a mis-indexed
gather, a dropped value, a sign flip), not floating-point noise.

## Task

Implement `dense_vs_compressed24_matmul_error(W, X)`:

```python
import numpy as np

def dense_vs_compressed24_matmul_error(W: np.ndarray, X: np.ndarray) -> float:
    ...
```

- `W`: `(m, n)` float64, **already** exactly 2:4 sparse — every
  consecutive group of 4 columns in every row has exactly 2 nonzero
  entries. `n` is divisible by `4`.
- `X`: `(n, p)` float64.

Compute two results and return the largest absolute elementwise
difference between them:

1. **Dense path**: `Y_dense = W @ X`, computed directly.
2. **Compressed path**: for every row and every group of 4 columns, read
   off the 2 nonzero values (in left-to-right order) and their in-group
   position (`0..3`); scatter them into a fresh all-zero `(m, n)` buffer
   at `group_start + position`; then matmul that reconstructed matrix
   against `X` to get `Y_compressed`.

Return `float(np.max(np.abs(Y_dense - Y_compressed)))`.

## Example

```python
import numpy as np
W = np.array([[4.0, 0.0, 0.0, 3.0]])   # one group of 4, already 2:4 sparse
X = np.array([[1.0], [1.0], [1.0], [1.0]])
dense_vs_compressed24_matmul_error(W, X)
# 0.0 -- both paths compute 4.0*1 + 3.0*1 = 7.0
```

## What the gate checks

The grader loads committed fixtures `w24.npy` (`(20, 32)`, already exactly
2:4 sparse in every group of 4 columns) and `x.npy` (`(32, 5)`), and
independently computes both paths with a NumPy oracle to get its own
reference discrepancy value.

The gate metric is `max_abs_err`, the absolute difference between your
returned value and the oracle's; it must be `< 1e-6`. Both paths compute
the *same* underlying arithmetic (the compressed path is exact, not
approximate), so a correct implementation returns a value at or near
machine epsilon and matches the oracle just as tightly. Mis-scattering a
value to the wrong in-group index, dropping a group, or reading indices
in the wrong order will make the compressed path silently compute a
different matrix — and the returned discrepancy will be far larger than
the gate's tolerance.
