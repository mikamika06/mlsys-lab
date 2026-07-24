## Context

Weight-rounding optimizers (GPTQ-style fine-tuning, OmniQuant, and
sign-gradient / SignSGD quantization-aware training loops) all need a
gradient *through* a `round()` — a function whose true derivative is
zero almost everywhere. The **straight-through estimator (STE)**
fixes this by pretending, only in the backward pass, that `round(x)`
is the identity function — except where clipping actually saturates,
where the gradient is defined to be exactly zero (clipping really does
kill the signal).

Let $V \in \mathbb{R}^{O\times I}$ be a learnable weight parameter, and
its quantized form (per-row scale $\text{scale}_o$, fixed, not
differentiated through, $q_{\max}=2^{\text{bits}-1}-1$):

$$
r_{o,i} = \frac{V_{o,i}}{\text{scale}_o}, \qquad
W^q_{o,i} = \text{scale}_o \cdot \mathrm{clip}\big(\mathrm{round}(r_{o,i}),\, -q_{\max},\, q_{\max}\big)
$$

Given activations $X\in\mathbb{R}^{B\times I}$ and the original
full-precision weight $W\in\mathbb{R}^{O\times I}$, the block MSE loss
being minimized is

$$
L(V) = \frac{1}{B\cdot O}\left\lVert XW^\top - XW^{q\top}\right\rVert_F^2
$$

With the STE convention above, $\partial W^q_{o,i}/\partial V_{o,i}
\approx m_{o,i}$ where $m_{o,i} = \mathbf{1}\big[|r_{o,i}| \le q_{\max}+0.5\big]$
(1 where rounding didn't need to clip, 0 where it did), and the chain
rule gives

$$
\frac{\partial L}{\partial V_{o,i}} = m_{o,i}\cdot\frac{2}{B\cdot O}\Big[(XW^{q\top} - XW^\top)^\top X\Big]_{o,i}
$$

This is exactly the gradient array a SignSGD-style optimizer consumes
(it only ever looks at its sign) to update $V$.

## Task

Implement `ste_block_mse_grad_wrt_v`:

```python
def ste_block_mse_grad_wrt_v(X: np.ndarray, W: np.ndarray, V: np.ndarray,
                              scale: np.ndarray, bits: int) -> np.ndarray:
    ...
```

- `X`: `(B, I)` `float64` activations.
- `W`: `(O, I)` `float64` original full-precision weight (the target).
- `V`: `(O, I)` `float64` current learnable weight parameter.
- `scale`: `(O,)` `float64`, positive, one fixed scale per output row.
- `bits`: quantizer bit width.

1. Compute $r = V / \text{scale}[:,\text{None}]$ and the mask
   $m = \mathbf{1}[\,|r| \le q_{\max}+0.5\,]$.
2. Compute $W^q$ (formula above), then
   $\text{pred} = X W^{q\top}$, $\text{target} = XW^\top$,
   $\text{diff} = \text{pred} - \text{target}$.
3. Return $m \odot \dfrac{2}{B\cdot O}\,\text{diff}^\top X$, shape `(O, I)`.

## Example

```python
import numpy as np
X = np.random.default_rng(0).standard_normal((8, 6))
W = np.random.default_rng(1).standard_normal((4, 6))
V = W.copy()          # start exactly at the target weight
scale = np.full(4, 0.3)
grad = ste_block_mse_grad_wrt_v(X, W, V, scale, bits=4)
# V == W here, but W != Wq(V) in general (rounding still perturbs it),
# so grad is generally nonzero even at V == W.
```

## What the gate checks

The grader builds several seeded `(X, W, V, scale, bits)` cases and
computes the reference gradient independently in NumPy with the exact
closed-form STE formula above (same mask definition, same `2/(B*O)`
normalization).

Note this is **not** checked against a numerical (finite-difference)
gradient — by design, `round()`'s true derivative is zero almost
everywhere, so a real finite-difference estimate of `dL/dV` is zero on
nearly every entry. STE is a deliberate, defined *substitute* gradient,
not an approximation of the true one, so the oracle here is the STE
formula itself. `rel_err` is the global relative L2 error between your
returned gradient array and the oracle's, across all cases (must be
`<= 1e-5`) — a wrong mask condition, a missing `2/(B*O)` factor, or a
transposed `diff.T @ X` all produce a large, visible error.
