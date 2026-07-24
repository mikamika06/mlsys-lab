## Context

GPTQ's Optimal Brain Quantization (OBQ) step doesn't just round each
weight column independently — it quantizes columns one at a time, **left
to right**, and after each column, propagates the rounding error into
every *not-yet-quantized* column to the right, using the inverse of the
layer's (Cholesky-derived) Hessian $H^{-1}$. This is what lets GPTQ
reach much lower error than naive per-column rounding: every remaining
column gets a small correction that compensates for exactly the error
just introduced.

For weight matrix $W \in \mathbb{R}^{r \times n}$ (rows = output
channels, columns = input features) and precomputed
$H^{-1} \in \mathbb{R}^{n \times n}$, quantizing column $i$ is:

$$
q_i = \mathrm{quantize}(w_i), \qquad
\mathrm{err} = \frac{w_i - q_i}{[H^{-1}]_{ii}}
$$

$$
W_{:, i+1:} \;\mathrel{-}= \; \mathrm{err} \otimes [H^{-1}]_{i,\, i+1:}
$$

($\otimes$ is the outer product: `err` is $r$-dimensional, `[H^{-1}]_{i,i+1:}`
is the row of remaining columns, so the update is a rank-1 correction to
every not-yet-processed column, scaled by how correlated that column is
with column $i$ under the Hessian).

## Task

Implement `obq_column_step`, a single step of this procedure:

```python
def obq_column_step(
    W: np.ndarray, H_inv: np.ndarray, col: int, scale: np.ndarray, nmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `W`: `(rows, n)` float64 weight matrix.
- `H_inv`: `(n, n)` float64 symmetric positive-definite inverse-Hessian
  matrix.
- `col`: int, the index of the column to quantize this step
  (`0 <= col < n`).
- `scale`: `(rows,)` float64, one symmetric quantization scale per row,
  for this column: `quantize(w) = clip(round(w / scale), -nmax, nmax) * scale`.
- `nmax`: positive int.

Return `(q_col, W_updated)`:

- `q_col`: `(rows,)`, the dequantized quantized column `col`, per the
  `quantize` formula above.
- `W_updated`: `(rows, n)`, equal to `W` except: column `col` is replaced
  by `q_col`, and every column after `col` (columns `col+1 .. n-1`) gets
  the rank-1 correction
  `W[:, col+1:] -= outer(err, H_inv[col, col+1:])` with
  `err = (W[:, col] - q_col) / H_inv[col, col]`. Columns before `col` are
  left untouched (in this single-step model, they were already finalized
  by earlier steps).

## Example

```python
import numpy as np

W = np.array([[1.3, 2.1, -0.4], [0.6, -1.8, 2.2]])
H_inv = np.array([[2.0, 0.5, -0.3], [0.5, 1.5, 0.2], [-0.3, 0.2, 1.0]])
scale = np.array([0.5, 0.5])

q_col, W_updated = obq_column_step(W, H_inv, col=0, scale=scale, nmax=3)
# q_col = clip(round(W[:,0]/0.5), -3, 3) * 0.5
# err = (W[:,0] - q_col) / H_inv[0,0]
# W_updated[:,0] = q_col; W_updated[:,1:] = W[:,1:] - outer(err, H_inv[0,1:])
```

## What the gate checks

The grader builds several `(W, H_inv, col, scale, nmax)` scenarios: a
seeded NumPy generator produces `W` and a genuine positive-definite
`H_inv` (built as the inverse of `A^T A + eps*I` for a random `A`, so
it's real, never hardcoded), varies `col` across the matrix (including
the last column, where no correction applies), and varies `scale` and
`nmax`. It computes the reference `(q_col, W_updated)` independently in
NumPy from the formulas above, never calling your function.

`rel_err` is the worst-case relative L2 error between your `W_updated`
(flattened, which also covers `q_col` since it becomes column `col` of
`W_updated`) and the oracle's, across every scenario, and the gate
requires `<= 1e-8`. Dividing by the wrong Hessian entry (e.g. using `1`
instead of `H_inv[col, col]`), propagating the correction to columns
before `col` instead of strictly after, using `+=` instead of `-=`, or
forgetting to also overwrite column `col` itself with `q_col` in
`W_updated` will all produce a visible mismatch.
