## Context

Pure magnitude pruning judges a weight by $|W_{ij}|$ alone — it has no
idea that some input features (activation channels) carry far more
signal energy than others. **Wanda** fixes this cheaply, with no
retraining and no Hessian: weight $|W_{ij}|$ times the $\ell_2$ norm of
the activations flowing through that input feature,
$\lVert X_{j,:}\rVert_2$. A weight on a "loud" channel is protected even
if it's individually small, because pruning it would silently discard a
lot of signal. This task verifies the claim directly: **at the same
sparsity, on the same calibration data, Wanda's pruning mask must not
lose more output energy than the pure-magnitude mask.**

### Setup

Given $W \in \mathbb{R}^{d_{out}\times d_{in}}$ and calibration
activations $X \in \mathbb{R}^{d_{in}\times n}$ (input features as rows),
and a sparsity fraction $\rho$:

* **Magnitude mask** $M_{\text{mag}}$: per output row $i$, zero the
  lowest-$\rho$-fraction of $|W_{i,:}|$ (stable sort, ties broken by
  column index).
* **Wanda mask** $M_{\text{wanda}}$: per output row $i$, zero the
  lowest-$\rho$-fraction of the importance score
  $|W_{i,j}|\cdot\lVert X_{j,:}\rVert_2$ (same stable-sort tie-break).

For either mask $M$, the output-reconstruction error is the squared
Frobenius norm of the output difference:

$$
e(M) = \lVert WX - (W\odot M)X \rVert_F^2
$$

## Task

Implement `wanda_vs_magnitude_error`:

```python
def wanda_vs_magnitude_error(W: np.ndarray, X: np.ndarray, sparsity: float) -> tuple[float, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `X` — `(d_in, n)` calibration activations.
* `sparsity` — fraction of each row's entries to prune, in `[0, 1)`.

Return `(e_wanda, e_magnitude)`, each computed exactly as defined above.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(16, 25))
X[3, :] *= 15.0  # one loud channel
W = rng.normal(size=(6, 16))
e_wanda, e_magnitude = wanda_vs_magnitude_error(W, X, sparsity=0.5)
assert e_wanda <= e_magnitude
```

## What the gate checks

* **wanda_rel_err** — relative error between your `e_wanda` and a NumPy
  oracle running the recipe above, over several random `(W, X)` trials
  with outlier activation channels.
* **magnitude_rel_err** — relative error between your `e_magnitude` and
  the oracle's, on the same trials.
* **wanda_le_magnitude** — your own `e_wanda` must never exceed your own
  `e_magnitude` (up to a small floating-point slack), on every trial.
