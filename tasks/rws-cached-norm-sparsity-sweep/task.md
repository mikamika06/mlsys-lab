## Context

Wanda ("Pruning by Weights and Activations") scores each weight not
just by its own magnitude but by how much it actually moves the
layer's output, using a calibration activation matrix
$X \in \mathbb{R}^{n\times d_{\text{in}}}$:

$$
S_{o,i} = |W_{o,i}| \cdot \lVert X_{:,i}\rVert_2
$$

where $\lVert X_{:,i}\rVert_2$ is the L2 norm of input feature $i$
across the $n$ calibration samples — an input channel that's almost
always near zero can't matter much no matter how large its weight is.
Pruning is done **per output row**, independently: for a target
sparsity $s\in(0,1)$, the $\lfloor s\cdot d_{\text{in}} + 0.5\rfloor$
lowest-scoring weights in that row are zeroed out (kept-mask $=0$),
the rest kept (mask $=1$).

The per-column norms $\lVert X_{:,i}\rVert_2$ don't depend on the
target sparsity at all — sweeping several sparsities (e.g. to pick the
best accuracy/compression trade-off) should compute that norm pass
**once** and reuse it for every sparsity level, rather than
recomputing it per sparsity.

## Task

Implement `wanda_masks_for_sparsities`:

```python
def wanda_masks_for_sparsities(W: np.ndarray, X: np.ndarray, sparsities: list[float]) -> list[np.ndarray]:
    ...
```

- `W`: `(out_features, in_features)` `float64` weight matrix.
- `X`: `(n_samples, in_features)` `float64` calibration activations.
- `sparsities`: list of target sparsity fractions in `(0, 1)`.

1. Compute the per-input-feature column norm once:
   `col_norm[i] = ||X[:, i]||_2`.
2. Compute the score matrix `S = |W| * col_norm[None, :]`.
3. For each sparsity `s` in `sparsities` (reusing `S` from step 2 —
   don't recompute norms per sparsity), independently per output row:
   `n_prune = round(s * in_features)`; rank that row's scores and zero
   out (mask `= 0`) the `n_prune` **lowest**-scoring entries, keep
   (mask `= 1`) the rest. Break ties by lowest column index first
   (`np.argsort` is stable, ascending — use it directly on the row's
   scores and take the first `n_prune` indices to prune).

Return a `list` of masks, one per entry of `sparsities` in the same
order, each `(out_features, in_features)`, entries `0` or `1`.

## Example

```python
import numpy as np
W = np.array([[1.0, -5.0, 0.2, 3.0]])
X = np.ones((10, 4))
X[:, 2] *= 100.0     # column 2 has a huge activation norm despite a tiny weight
masks = wanda_masks_for_sparsities(W, X, [0.5])
# scores ~= [1.0*sqrt(10), 5.0*sqrt(10), 0.2*(100*sqrt(10)), 3.0*sqrt(10)]
# column 2's tiny weight now scores highest (loud activation), column 1
# (the numerically largest weight) can still get pruned if its activation
# is quiet -- this is exactly what distinguishes Wanda from magnitude pruning.
```

## What the gate checks

The grader builds several seeded `(W, X, sparsities)` cases (continuous
random values, so exact score ties essentially never occur) and computes
every mask independently in NumPy with the same score formula, the same
per-row `round(s * in_features)` prune count, and the same stable
ascending-argsort tie-break.

`exact_match` is the fraction of mask entries, across every sparsity and
every case, that match the oracle exactly (must equal `1.0`) — any
mismatch in the score formula (e.g. forgetting the activation norm and
falling back to plain magnitude pruning), the per-row vs. global ranking
axis, or the prune-count rounding convention shows up as a large,
visible drop from `1.0`.
