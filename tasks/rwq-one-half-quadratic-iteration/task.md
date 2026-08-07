## Context

**HQQ** (Half-Quadratic Quantization) tunes a group's zero-point $z$ to
minimize an $\ell_p$-norm ($p \le 1$, a robust, outlier-tolerant loss)
of the rounding residual instead of the usual min-max heuristic. Because
the $\ell_p$ objective isn't directly minimizable in closed form, HQQ uses
**half-quadratic splitting**: introduce an auxiliary residual variable,
alternately (a) shrink it towards the current residual with a
closed-form generalized shrinkage/soft-threshold operator, then (b)
re-solve for $z$ in closed form (a simple mean) treating the shrunk
residual as fixed. This task implements **one such iteration**.

### Setup

Weights $W \in \mathbb{R}^{d_{out}\times d_{in}}$ are grouped **per row**.
Each row has a scalar scale $s_i$ and zero-point $z_i$, plus the row's
current integer codes $W_{q,i} \in \{q_{\min},\dots,q_{\max}\}^{d_{in}}$.

### One iteration (per row, broadcasting over columns)

1. **Target before rounding**: $\mathrm{raw} = W/s + z$.
2. **Residual** against the given codes: $r = W_q - \mathrm{raw}$.
3. **Generalized-$\ell_p$ shrinkage** of the residual, with strength
   $\beta$ and norm order $p$:
   $$
   \mathrm{shrink}(x,\beta,p) =
   \begin{cases}
   \mathrm{sign}(x)\cdot\mathrm{relu}\!\left(|x| - \dfrac{1}{\beta}\right) & p = 1 \\[6pt]
   \mathrm{sign}(x)\cdot\mathrm{relu}\!\left(|x| - \dfrac{1}{\beta}(|x|+\varepsilon)^{\,p-1}\right) & p \ne 1
   \end{cases}
   \qquad \varepsilon = 10^{-8}
   $$
   applied elementwise: $W_e = \mathrm{shrink}(r,\beta,p)$.
4. **Zero-point update** — the least-squares $z$ that makes
   $W_q - W/s - z \approx W_e$, averaged over the row (the group):
   $$
   z_{\text{new}} = \mathrm{mean}_{\text{row}}\!\left(W_q - W_e - W/s\right)
   $$
5. **Re-quantize** with the new zero-point:
   $$
   W_{q,\text{new}} = \mathrm{clip}\!\left(\mathrm{round}(W/s + z_{\text{new}}),\, q_{\min},\, q_{\max}\right)
   $$

## Task

Implement `hqq_half_quadratic_step`:

```python
def hqq_half_quadratic_step(W: list[list[float]], s: list[float], z: list[float], W_q: list[list[float]], lp: float, beta: float, qmin: int, qmax: int) -> tuple[list[list[float]], list[float]]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `s`, `z` — `(d_out,)`, per-row scale and current zero-point.
* `W_q` — `(d_out, d_in)`, the current integer codes (given, used as-is in
  steps 2 and 4 — do **not** recompute it before step 5).
* `lp` — the $\ell_p$ norm order (e.g. `0.7`, `1.0`).
* `beta` — shrinkage strength.
* `qmin`, `qmax` — integer clip bounds for the codes.

Return `(W_q_new, z_new)` — the re-quantized codes (step 5) and the
updated zero-point (step 4), both computed exactly as above.

## Example

```python
d_out, d_in = 4, 16
W = [[random.gauss(0, 1) for _ in range(d_in)] for _ in range(d_out)]
s = [0.3] * d_out
z = [8.0] * d_out
W_q = [[max(0, min(15, round(W[i][j] / s[i] + z[i]))) for j in range(len(W[0]))] for i in range(len(W))]
W_q_new, z_new = hqq_half_quadratic_step(W, s, z, W_q, lp=0.7, beta=10.0, qmin=0, qmax=15)
```

## What the gate checks

* **z_max_abs_err** — max-abs difference between your `z_new` and a
  Python oracle running steps 1-4 above, over several random states and
  `(lp, beta)` settings.
* **wq_exact_match** — your `W_q_new` must exactly equal the oracle's
  re-quantized codes (step 5) on every trial.
