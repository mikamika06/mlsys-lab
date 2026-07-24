## Context

**2:4 structured sparsity** requires that every consecutive group of 4
weights in a row keep exactly 2 nonzero entries (a pattern many
accelerators can exploit for a real 2x speedup). Two very different ways
to pick which 2 to keep:

**Magnitude pruning** looks only at each weight's own size: for each
group of 4, zero the 2 smallest-$|w|$ entries, keep the 2 largest,
unmodified. It never looks at the activations at all.

**SparseGPT** uses a local second-order (Hessian) approximation of how
much removing a weight actually hurts the *layer output*, given
calibration activations $X\in\mathbb{R}^{s\times n}$:

$$
H = \frac{X^\top X}{s} + \lambda I, \qquad \lambda = 10^{-4}.
$$

For weight $w_{ij}$, its saliency is $S_{ij} = w_{ij}^2 / (H^{-1})_{jj}$ —
a weight sitting on a highly-activated or highly-correlated input channel
gets a large $(H^{-1})_{jj}^{-1}$-type penalty even if $|w_{ij}|$ itself
is small. Within each group of 4, the 2 lowest-saliency weights are
pruned; crucially, the 2 **kept** weights are then compensated using the
inverse-Hessian: pruning $w_{ij}$ updates every still-kept weight $w_{ik}$
in the same group,

$$
w_{ik} \leftarrow w_{ik} - w_{ij}\,\frac{(H^{-1})_{kj}}{(H^{-1})_{jj}}.
$$

Magnitude pruning is blind to activation statistics, so it can happily
prune a small-but-important weight on a salient or correlated channel;
SparseGPT's saliency score and compensation are built specifically to
avoid that.

## Task

Implement `compare_magnitude_vs_sparsegpt_2_4(W, X)`:

```python
def compare_magnitude_vs_sparsegpt_2_4(W: np.ndarray, X: np.ndarray) -> tuple[float, float, float]:
    ...
```

- `W`: `(out_dim, in_dim)` float weight matrix; `in_dim` is a multiple of 4.
- `X`: `(s, in_dim)` float calibration activations.

1. `Y_true = X @ W.T`.
2. **Magnitude 2:4**: prune as described above (no compensation);
   `err_magnitude` = relative Frobenius output error
   `||X @ W_hat_mag.T - Y_true||_F / ||Y_true||_F`.
3. **SparseGPT 2:4**: prune + compensate as described above (damping
   $\lambda=10^{-4}$); `err_sparsegpt` = the same relative Frobenius
   output error using the compensated weights.
4. `reduction = 1 - err_sparsegpt / err_magnitude`.

Return `(err_magnitude, err_sparsegpt, reduction)`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(6, 8))
A = rng.normal(size=(8, 8)) * 0.3 + np.eye(8)
X = (rng.normal(size=(40, 8)) @ A)
X[:, [1, 5]] *= 12.0   # channels 1 and 5 are salient/correlated

err_magnitude, err_sparsegpt, reduction = compare_magnitude_vs_sparsegpt_2_4(W, X)
# err_sparsegpt is noticeably smaller than err_magnitude; reduction > 0
```

## What the gate checks

The gate loads a fixed fixture (`W.npy`: 8x16, `X.npy`: 64x16 correlated
activations with 3 salient channels scaled 15x) plus several seeded
synthetic `(W, X)` pairs built the same way (correlated + salient
channels — i.i.d. random activations wouldn't reliably show SparseGPT's
advantage, since its edge comes specifically from exploiting activation
structure magnitude pruning ignores). For each, the oracle independently
recomputes `(err_magnitude, err_sparsegpt, reduction)` with the exact
formulas above.

Your returned triple is compared to the oracle's with the `rel_err`
scorer (relative L2 error over the 3-vector), and the worst case across
every scenario must be `< 1e-6`. Skipping the inverse-Hessian
compensation step (pruning without updating the kept weights), scoring
saliency by `|w|` instead of `w^2 / Hinv[j,j]`, dividing by `Hinv[k,k]`
instead of `Hinv[j,j]` in the compensation update, or forgetting the
damping term `lambda * I` before inverting (an ill-conditioned or
singular `H`) will all miss the tolerance.
