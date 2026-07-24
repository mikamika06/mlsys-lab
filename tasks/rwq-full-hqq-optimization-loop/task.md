## Context

Standard affine quantization picks scale $s$ and zero-point $z$ from the
tensor's min/max and rounds every element the same way, so a single
outlier stretches the whole range and wastes precision on everything
else. **HQQ** (Half-Quadratic Quantization) keeps $s$ fixed (from the
usual min/max formula) but *optimizes* $z$ against a robust $L_p$
($p<2$) reconstruction loss instead of the min/max choice, using a
half-quadratic splitting iteration — the same family of solver used for
robust ("edge-preserving") signal reconstruction. This lets a handful of
iterations push $z$ towards the value that best fits the bulk of the
weights, largely ignoring the outliers, without ever touching $s$ or
paying for gradient descent.

### The loop

Given weights $W$, fixed scale $s$, an initial zero-point $z_0$,
quantization bounds $[q_{min}, q_{max}]$, an $L_p$ exponent $p \in (0,2]$,
an initial penalty $\beta_0$, growth factor $\kappa$, and iteration count
$T$, HQQ repeats, for $t = 0, \dots, T-1$:

$$
W_q = \mathrm{clip}(\mathrm{round}(sW + z_t),\ q_{min},\ q_{max})
\qquad
W_r = \frac{W_q - z_t}{s}
$$

$$
W_e = \mathrm{shrink}(W - W_r,\ \beta_t, p)
\qquad
z_{t+1} = \mathrm{mean}\big(W_q - (W - W_e)\,s\big)
\qquad
\beta_{t+1} = \beta_t \cdot \kappa
$$

where the shrink operator is the proximal map of $\beta^{-1}\lVert\cdot\rVert_p^p$:

$$
\mathrm{shrink}(x,\beta,p) = \mathrm{sign}(x)\cdot\max\!\left(|x| - \tfrac{1}{\beta}|x|^{p-1},\ 0\right)
\quad\text{(for } p=1,\ \tfrac{1}{\beta}|x|^{p-1}=\tfrac1\beta\text{, ordinary soft-threshold)}.
$$

After the $T$ passes above (which only ever update $z$), quantize **once
more** with the converged $z_T$ to get the values actually returned:

$$
W_q^{\text{final}} = \mathrm{clip}(\mathrm{round}(sW + z_T),\ q_{min},\ q_{max}),
\qquad
\hat W = \frac{W_q^{\text{final}} - z_T}{s}.
$$

## Task

Implement:

```python
def hqq_optimize(W, scale, zero0, qmin, qmax, lp_norm, beta0, kappa, iters):
    ...
```

* `W` — 1-D `float64` array of weights (one HQQ group).
* `scale` — fixed scalar $s$ (already computed, e.g. from min/max — do not
  recompute or change it).
* `zero0` — initial scalar zero-point $z_0$.
* `qmin`, `qmax` — integer quantization bounds (e.g. `0`, `2**nbits - 1`).
* `lp_norm` — exponent $p$ for the shrink operator.
* `beta0`, `kappa` — initial penalty and its per-iteration growth factor.
* `iters` — number of half-quadratic passes $T$ (run exactly this many,
  no early stopping — determinism matters for grading).

Return `(W_q, z, W_dequant)`:

* `W_q` — the **final** quantized integer codes (as an array), from
  re-quantizing $W$ with the converged $z_T$ (not from inside the loop).
* `z` — the converged scalar zero-point $z_T$.
* `W_dequant` — $(W_q - z) / s$.

## Example

```python
import numpy as np
W = np.array([0.1, -0.2, 0.05, 4.0, -0.1])  # 4.0 is an outlier
scale = 2.0
zero0 = 0.0
Wq, z, Wdq = hqq_optimize(W, scale, zero0, qmin=0, qmax=15,
                           lp_norm=0.7, beta0=1.0, kappa=1.5, iters=10)
# z converges toward a value that fits the bulk (~[-0.2, 0.1]) well,
# largely ignoring the 4.0 outlier's pull on a min/max zero-point.
```

## What the gate checks

* **exact_match** — your final `W_q` (integer codes) must equal, element
  for element, an oracle that runs the identical loop above (same
  `scale`, `zero0`, shrink operator, and re-quantization step) in NumPy,
  on several random weight vectors (with an injected outlier) and
  hyperparameter settings, seeded for determinism.
* **max_abs_err** — the maximum absolute difference between your
  `W_dequant` and the oracle's, on the same cases, must be $\le 10^{-6}$.
