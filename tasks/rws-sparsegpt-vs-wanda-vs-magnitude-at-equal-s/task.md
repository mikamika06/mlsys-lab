## Context

Three one-shot pruning methods, all targeting the **same** global sparsity
$s$ on a weight matrix $W \in \mathbb{R}^{m\times d}$ with calibration
activations $X \in \mathbb{R}^{d\times n}$ ($d$ input features, $n$
calibration samples), differ in how much information they use to decide
*which* weights to drop and whether they *compensate* the survivors:

- **Magnitude**: drop the globally smallest-$|W_{ij}|$ entries. No
  activation information, no compensation.
- **Wanda**: drop the globally smallest $|W_{ij}|\cdot z_j$ entries, where
  $z_j = \lVert X_{j,:}\rVert_2$ is input feature $j$'s activation norm
  across the calibration set. Activation-aware, still no compensation.
- **SparseGPT** (OBS-based): using
  $H = 2XX^\top + \lambda I \in \mathbb{R}^{d\times d}$, drop the globally
  smallest $w_{ij}^2 / [H^{-1}]_{jj}$ entries, and after dropping each
  weight, **compensate** the rest of that same output row:

$$
W_{i,:} \mathrel{+{=}} -\,w_{ij}\,\frac{[H^{-1}]_{j,:}}{[H^{-1}]_{jj}}, \qquad\text{then } W_{i,j}=0.
$$

(Weights are dropped one at a time, in ascending score order, applying
the compensation update after each removal — exactly the OBS procedure.)

For all three methods, $\texttt{remove} = \lfloor m d\, s\rfloor$ weights
are dropped in total (a shared global count, so all three land on
*exactly* the same overall sparsity $s$). The **layer reconstruction
MSE** of a candidate $\hat W$ is

$$
\mathrm{mse}(\hat W) = \frac{1}{m n}\bigl\lVert WX - \hat WX \bigr\rVert_F^2 .
$$

Because SparseGPT uses more information (activations *and* Hessian-based
compensation) than Wanda (activations only, no compensation), which in
turn uses more information than magnitude (neither), the reconstruction
error should satisfy $\mathrm{mse}(\hat W_{\text{SparseGPT}}) \le
\mathrm{mse}(\hat W_{\text{Wanda}}) \le \mathrm{mse}(\hat
W_{\text{magnitude}})$ at equal sparsity.

## Task

Implement `compare_prune_methods_mse(W, X, sparsity, lam)`:

```python
import numpy as np

def compare_prune_methods_mse(W: np.ndarray, X: np.ndarray, sparsity: float, lam: float) -> dict:
    ...
```

- `W`: `(m, d)` float64 weight matrix.
- `X`: `(d, n)` float64 calibration activations.
- `sparsity`: float in `(0, 1)`, the shared target global sparsity.
- `lam`: float damping added to the SparseGPT Hessian diagonal.

Compute all three pruned reconstructions as defined above and return:

```python
{"mse_magnitude": float, "mse_wanda": float, "mse_sparsegpt": float}
```

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
W = rng.standard_normal((6, 8))
X = rng.standard_normal((8, 32))
out = compare_prune_methods_mse(W, X, sparsity=0.5, lam=0.01)
out["mse_sparsegpt"] <= out["mse_wanda"] <= out["mse_magnitude"]   # True
```

## What the gate checks

The grader loads committed fixtures `layer_w.npy` (`(10, 16)`) and
`layer_x.npy` (`(16, 64)`, correlated features with varying per-feature
scale so activation-aware pruning genuinely differs from plain
magnitude), and recomputes all three reconstructions independently in
NumPy at `sparsity = 0.5`, `lam = 0.01`.

- `magnitude_err`, `wanda_err`, `sparsegpt_err`: relative error between
  your returned MSE for each method and the oracle's; each gate `< 1e-6`.
- `ordering_ok`: `1.0` if **your own** three returned values satisfy
  `mse_sparsegpt <= mse_wanda <= mse_magnitude`, else `0.0`.

Dropping the Wanda activation-norm factor (reducing it to magnitude
pruning), forgetting the OBS compensation step (reducing SparseGPT to a
Hessian-weighted magnitude prune), or pruning per-row instead of globally
(breaking the shared `remove` count that keeps all three at exactly the
same overall sparsity) will miss one or more gates.
