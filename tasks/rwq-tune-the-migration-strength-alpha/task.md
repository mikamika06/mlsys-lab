## Context

Activations in transformer layers often have a handful of extreme-outlier
channels while weights stay well-behaved — quantizing both to int8
independently punishes activations badly. **SmoothQuant** migrates
quantization difficulty from activations to weights with a per-input-channel
scale $s_j$, controlled by a strength $\alpha \in [0,1]$:

$$
s_j = \frac{\max_i |X_{i,j}|^{\alpha}}{\max_r |W_{r,j}|^{1-\alpha}}
$$

$\alpha=0$ leaves activations alone (all difficulty stays on outlier
channels); $\alpha=1$ pushes the *entire* activation range onto the
weights. Neither extreme is optimal when activations have real outliers
*and* weights aren't perfectly flat — the best $\alpha$ is a middle
ground, found by sweeping a grid and measuring actual **W8A8** (int8
weights, int8 activations) output error.

### One evaluation at a fixed alpha

1. **Smooth**: $X' = X \,\mathrm{diag}(s)^{-1}$, $W' = W\,\mathrm{diag}(s)$
   (each column $j$ of $X$ divided by $s_j$, each column $j$ of $W$
   multiplied by $s_j$ — this leaves $X W^\top$ mathematically unchanged
   before quantization).
2. **Quantize activations**: one dynamic, per-tensor symmetric int8 scale,
   $s_x = \max|X'|/127$; $X'_q = \mathrm{clip}(\mathrm{round}(X'/s_x),-127,127)\cdot s_x$.
3. **Quantize weights**: one symmetric int8 scale *per output row* $r$,
   $s_{w,r} = \max_j|W'_{r,j}|/127$; $W'_q$ analogous to above.
4. **Output MSE**: $\mathrm{mse}(\alpha) = \frac{1}{n\, d_{out}}\lVert X'_qW'^\top_q - XW^\top\rVert_F^2$.

(Use $\epsilon=10^{-8}$ in the denominator of $s_j$ and floor $s_j$ at
$\epsilon$ to avoid division by zero; use scale $1$ wherever a max-abs is
exactly $0$.)

## Task

Implement `sweep_alpha`:

```python
def sweep_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray) -> tuple[int, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `X` — `(n_cal, d_in)` calibration activation matrix.
* `alphas` — 1-D array of candidate $\alpha$ values to sweep, in order.

Return `(best_idx, best_mse)`:

* `best_idx` — the index into `alphas` of the value minimizing
  $\mathrm{mse}(\alpha)$ as defined above.
* `best_mse` — that minimum MSE value.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(40, 12))
X[:, [2, 7]] *= 20.0  # outlier channels
W = rng.normal(size=(5, 12))
alphas = np.linspace(0.0, 1.0, 11)
best_idx, best_mse = sweep_alpha(W, X, alphas)
```

## What the gate checks

* **argmin_index_match** — your `best_idx` must exactly match a NumPy
  oracle that recomputes $s$, re-quantizes, and re-measures MSE for every
  alpha in the shared grid, on several random `(W, X)` trials with
  outlier activation channels.
* **mse_rel_err** — relative error between your `best_mse` and the
  oracle's minimum MSE, on the same trials.
