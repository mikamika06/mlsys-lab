## Context

Plain magnitude pruning ignores how a weight's removal actually affects
the layer's *output*. SparseGPT (and the Optimal Brain Surgeon /
Optimal Brain Compression line of work it builds on) instead uses the
**layer Hessian** $H = X^\top X$ (from calibration activations $X$) to
score and compensate for pruning:

**Mask selection.** A weight $w_q$'s contribution to the output error
if zeroed alone is $w_q^2 / [H^{-1}]_{qq}$ — cheap to compute for every
weight from just the diagonal of $H^{-1}$. For 2:4 structured sparsity,
every contiguous group of 4 weights (per row) keeps the 2 with the
**highest** score (equivalently prunes the 2 lowest):

$$
\text{score}(w_q) = \frac{w_q^2}{[H^{-1}]_{qq}}
$$

**Error-compensating update (OBS/OBC).** Zeroing a *set* of weight
indices $S$ in one shot (rather than one at a time) has a closed-form
optimal update for the surviving weights of that row, minimizing the
Hessian-weighted output error $\Delta w^\top H\,\Delta w$ subject to
$w'_S = 0$:

$$
w' = w - H^{-1}_{:,S}\,\big(H^{-1}_{S,S}\big)^{-1} w_S
$$

($w_S$ = the row's original values *at* the pruned positions,
$H^{-1}_{:,S}$ = those columns of $H^{-1}$, $H^{-1}_{S,S}$ = the
$|S|\times|S|$ submatrix). This provably drives $w'_S$ to exactly zero
while optimally nudging every surviving weight in the row to
compensate — cheaper 2nd-order pruning without retraining.

## Task

Implement `sparsegpt_24_prune`:

```python
def sparsegpt_24_prune(W: np.ndarray, X: np.ndarray, damp: float = 0.01):
    ...
```

- `W`: `(O, I)` `float64` weight matrix, `I % 4 == 0`.
- `X`: `(n, I)` `float64` calibration activations.
- `damp`: relative Hessian damping.

1. `H = X.T @ X`; add damping: `H += damp * mean(diag(H)) * I_identity`.
2. `Hinv = inv(H)`; `diag_hinv = diag(Hinv)` (shape `(I,)`, shared by
   every row — $H$ doesn't depend on the output row).
3. **Mask**: for every row and every contiguous group of 4 columns,
   `scores = W[row, group]**2 / diag_hinv[group]`; prune (`mask = 0`)
   the 2 lowest-scoring columns of that group, keep (`mask = 1`) the
   other 2.
4. **Compensation**: for each row independently, let `S` be that row's
   pruned column indices (`mask[row] == 0`, size `I/2`). Compute
   `w_S = W[row, S]`,
   `delta = -(Hinv[:, S] @ solve(Hinv[np.ix_(S, S)], w_S))`,
   `W_hat[row] = W[row] + delta`, then set `W_hat[row, S] = 0.0`
   exactly.

Return `(mask, W_hat)`: `mask` shape `(O, I)` with entries `0`/`1`,
`W_hat` shape `(O, I)` `float64`.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
W = rng.standard_normal((3, 16))
X = rng.standard_normal((50, 16))
mask, W_hat = sparsegpt_24_prune(W, X)
# every contiguous group of 4 columns in every row of `mask` has
# exactly two 1s and two 0s; W_hat is exactly zero at the pruned
# positions and otherwise close to (but not exactly) W, adjusted to
# compensate for the removed weights.
```

## What the gate checks

The grader builds several seeded `(W, X)` cases and computes the
reference mask and `W_hat` independently in NumPy with the exact
algorithm above (same Hessian, same diagonal score, same one-shot OBS
compensation formula).

`mask_exact_match` is `1.0` only if your `mask` matches the oracle's
exactly on every entry of every case (must equal `1.0`) — continuous
random weights make exact score ties essentially impossible, so any
mismatch means a real scoring or selection bug. `valid_24_fraction` is
the fraction of your *own* returned mask's 4-groups that contain
exactly two `1`s (must equal `1.0`) — a mask that happens to match the
oracle already satisfies this, but it catches a genuinely invalid
(non-2:4) mask independently of the exact-match comparison.
`what_rel_err` is the global relative L2 error between your `W_hat`
and the oracle's (must be `<= 1e-5`) — this catches a correct mask
paired with a wrong (or missing) OBS compensation update, e.g.
naively zeroing the pruned weights instead of applying the closed-form
correction.
