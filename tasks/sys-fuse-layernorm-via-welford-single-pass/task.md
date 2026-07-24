## Context

A naive LayerNorm forward pass scans each row twice: once to compute the
mean, and a second time to compute the variance around that mean. Fused
kernels avoid the second memory pass by computing the mean and variance
together, incrementally, while scanning the row **once**, using
**Welford's online algorithm**.

For a row $x = (x_1,\dots,x_D)$, Welford's recurrence visits elements in
order $j=1,\dots,D$ and maintains a running count, mean, and sum of
squared deviations $M_2$:

$$
\mathrm{count}_j = j, \qquad
\delta_j = x_j - \mathrm{mean}_{j-1}, \qquad
\mathrm{mean}_j = \mathrm{mean}_{j-1} + \frac{\delta_j}{j},
$$

$$
\delta_j' = x_j - \mathrm{mean}_j, \qquad
M_{2,j} = M_{2,j-1} + \delta_j \cdot \delta_j',
$$

starting from $\mathrm{mean}_0 = 0,\ M_{2,0} = 0$. After the single pass
over $j=1,\dots,D$,

$$
\mu = \mathrm{mean}_D, \qquad \sigma^2 = \frac{M_{2,D}}{D}
$$

are exactly the same population mean and (biased) variance that a
two-pass computation

$$
\mu = \frac{1}{D}\sum_{j=1}^{D} x_j, \qquad
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D} (x_j - \mu)^2
$$

would produce (up to floating-point rounding) — but Welford's version
never re-reads the row after the pass finishes.

LayerNorm then applies the usual normalize-and-affine step per row:

$$
\hat x_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_j = \gamma_j \, \hat x_j + \beta_j .
$$

## Task

Implement `layer_norm_welford`:

```python
def layer_norm_welford(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    ...
```

* `x` — 2-D array of shape $(B, D)$, `float64`.
* `gamma`, `beta` — 1-D arrays of shape $(D,)$, the affine parameters.
* `eps` — numerical-stability constant added to the variance.

For every row, compute the mean and variance with a **single explicit
pass over the $D$ feature columns** using Welford's online recurrence
(update a running `mean` and `M2` per row as you scan column
$j=0,1,\dots,D-1$ — the update at column $j$ must only use values already
seen, never look ahead and never re-scan earlier columns). Do **not**
compute the mean with one reduction and the variance with a second,
independent reduction (e.g. `x.mean(...)` followed by `x.var(...)` or
`((x - mean) ** 2).mean(...)`) — that is the two-pass approach this task
replaces.

Return the $(B, D)$ array `y` given by the normalize-and-affine formula
above. You may vectorize freely across the batch dimension $B$ inside the
per-column loop.

## Example

```python
import numpy as np

x = np.array([[1.0, 2.0, 3.0, 4.0]])
gamma = np.ones(4)
beta = np.zeros(4)

y = layer_norm_welford(x, gamma, beta)
# mean=2.5, var=1.25 (population) -> y ~= [-1.3416, -0.4472, 0.4472, 1.3416]
```

## What the gate checks

* **max_abs_err** — the grader builds several `(x, gamma, beta)` cases
  and compares your output element-wise against a standard two-pass
  NumPy reference (`x.mean(axis=1)`, `x.var(axis=1)`, then the affine
  transform). Must be $\le 10^{-5}$.
* **line_events** — using `sys.settrace`, the grader counts Python-level
  line executions inside your call. A genuine per-column Welford loop
  over $D$ columns emits many line events; a solution that secretly
  calls NumPy's own `mean`/`var` reductions (doing the forbidden two-pass
  computation in C) emits almost none. Must be $\ge 200$.
