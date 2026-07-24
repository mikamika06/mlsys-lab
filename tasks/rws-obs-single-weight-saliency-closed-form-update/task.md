## Context

**Optimal Brain Surgeon (OBS)** decides which weight to prune next not by
magnitude, but by the actual second-order cost of removing it — and it
doesn't just zero the weight, it **compensates every other weight** to
minimize the damage, using the loss's local curvature (Hessian $H$).

Assume the network is near a local loss minimum, so a second-order Taylor
expansion in a small perturbation $\delta w$ around the current weight
vector $w$ gives $\Delta L(\delta w) \approx \tfrac12 \delta w^\top H \delta w$
(the gradient term vanishes at a minimum). We want the smallest-cost
$\delta w$ that forces one specific coordinate $q$ to exactly zero:

$$
\min_{\delta w} \ \tfrac12 \delta w^\top H \delta w
\quad \text{s.t.} \quad e_q^\top \delta w + w_q = 0
$$

Solving this constrained quadratic (Lagrange multipliers) gives closed
forms for both the best $\delta w$ and the resulting loss increase:

$$
\delta w = -\frac{w_q}{[H^{-1}]_{qq}}\, H^{-1} e_q, \qquad
s_q = \frac{w_q^2}{[H^{-1}]_{qq}}
$$

$s_q$ (the **saliency** of coordinate $q$) is exactly twice the resulting
loss increase: $\Delta L = \tfrac12 s_q$. The weight to prune is whichever
coordinate has the **smallest** saliency — the least damage.

## Task

Implement `obs_prune_step`:

```python
def obs_prune_step(H: np.ndarray, w: np.ndarray) -> tuple[int, np.ndarray, float]:
    ...
```

* `H` — `(d, d)` symmetric positive-definite Hessian.
* `w` — `(d,)` current weight vector.

Return `(q, delta_w, dL)`:

* `q` — `int`, $\mathrm{argmin}_q\, s_q$ over all $d$ coordinates.
* `delta_w` — `(d,)` float array, the closed-form update above for the
  chosen `q` (so `w[q] + delta_w[q] == 0` exactly).
* `dL` — `float`, the analytic second-order loss change
  $\tfrac12\,\delta w^\top H\,\delta w$ from applying `delta_w`.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
d = 6
A = rng.normal(size=(d + 4, d))
H = A.T @ A + 0.1 * np.eye(d)  # SPD
w = rng.normal(size=d)
q, delta_w, dL = obs_prune_step(H, w)
assert abs(w[q] + delta_w[q]) < 1e-9
```

## What the gate checks

* **argmin_index_match** — your chosen `q` must exactly match a NumPy
  oracle's $\mathrm{argmin}_q s_q$, over several random `(H, w)` trials.
* **deltaw_max_abs_err** — max-abs difference between your `delta_w` and
  the oracle's closed-form update.
* **deltaL_rel_err** — relative error between your `dL` and the oracle's
  analytic loss change, on the same trials.
