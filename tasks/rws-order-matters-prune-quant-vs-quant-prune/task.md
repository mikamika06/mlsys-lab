## Context

A compression pipeline that both prunes and quantizes a layer can apply the two steps
in either order, and the order changes the result. Both orderings use the same
Wanda-style importance score to decide *which* weights survive,
$S_{ij} = |W_{ij}| \cdot \lVert X_{:,j} \rVert_2$, keeping the top
$k = \max(1, \mathrm{round}((1-\texttt{sparsity}) \cdot \texttt{group\_size}))$ per
group — but they differ in *when* the quantization scale is measured:

- **Prune-then-quant**: zero out the pruned weights first, then set the scale from the
  max magnitude of the *surviving* weights only:
$$
a_A = \max_{i \in \text{kept}} |w_i|, \qquad s_A = \frac{a_A}{q_{\max}}.
$$

- **Quant-then-prune**: set the scale from the max magnitude of the *entire, unpruned*
  group, quantize everything, and only then zero out the pruned positions:
$$
a_B = \max_i |w_i|, \qquad s_B = \frac{a_B}{q_{\max}}.
$$

If a weight with large raw magnitude but low importance score ends up pruned (its
activation column has little signal, so its score is low despite its size), it still
dominates $a_B$ — wasting quantization resolution on a value that gets thrown away
anyway. Under prune-then-quant that outlier never enters the scale computation, so the
surviving weights get a much finer quantization step. Pruning first (then quantizing the
survivors) therefore reconstructs the original tensor at least as accurately.

## Task

Implement `compare_prune_quant_order`:

```python
def compare_prune_quant_order(
    W: np.ndarray, X: np.ndarray, group_size: int, sparsity: float, bits: int = 4
) -> tuple[float, float]:
    ...
```

- `W`: `float64` array of shape `(rows, cols)`, `cols` an exact multiple of
  `group_size`.
- `X`: `float64` array of shape `(n_samples, cols)`, calibration activations.
- `group_size`: number of columns per quantization group / per-group Wanda ranking.
- `sparsity`: fraction of each group to prune.
- `bits`: symmetric quantization bit width, $q_{\max} = 2^{\text{bits}-1} - 1$.

For every group of `group_size` columns (within every row), compute the Wanda score,
determine the keep-set, then compute both orderings' dequantized reconstruction as
described above (pruned positions are `0` in both). Return
`(mse_prune_then_quant, mse_quant_then_prune)`: the mean squared error of each
ordering's full reconstruction against the original `W`, averaged over every element of
the whole matrix.

## Example

```python
import numpy as np

# One group: 15 small values, one huge-magnitude but zero-variance (unimportant) column.
w = np.full(16, 0.05)
w[0] = 1.5  # loud but, per X below, unimportant
X = np.ones((10, 16))
X[:, 0] = 1e-4  # column 0 carries almost no signal -> low Wanda score

mse_a, mse_b = compare_prune_quant_order(w.reshape(1, -1), X, group_size=16, sparsity=0.3)
# column 0 has the lowest score despite being the largest weight -> it gets pruned
# prune-then-quant: scale from the 11 surviving 0.05-magnitude weights -> fine step
# quant-then-prune: scale from amax=1.5 (the pruned outlier) -> coarse step wasted on nothing
# mse_a <= mse_b
```

## What the gate checks

The gate builds a NumPy oracle running the identical per-group pipeline on a fixed test
`W`/`X` pair, where each group has one loud-but-unimportant column engineered to make
the two orderings diverge. It checks:

- `mse_prune_then_quant_err`: absolute error of your `mse_prune_then_quant` vs. the
  oracle's, at most $10^{-9}$.
- `mse_quant_then_prune_err`: absolute error of your `mse_quant_then_prune` vs. the
  oracle's, at most $10^{-9}$.
- `order_correct`: your own `mse_prune_then_quant` must be `<=` your own
  `mse_quant_then_prune` (must be `1.0`).
