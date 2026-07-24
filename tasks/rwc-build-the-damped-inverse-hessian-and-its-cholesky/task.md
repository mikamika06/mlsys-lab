## Context

GPTQ quantizes a linear layer's weight matrix one column at a time,
compensating later columns for the error introduced by earlier ones,
using second-order information from calibration activations $X \in
\mathbb{R}^{n_{\text{cal}}\times d_{\text{in}}}$. The layer's Hessian
proxy (from a squared-error objective) is

$$
H = 2\,X^\top X
$$

Directly inverting $H$ is numerically fragile (it can be near-singular),
so GPTQ **damps** the diagonal first, proportional to the mean diagonal
magnitude:

$$
H_{\text{damped}} = H + \lambda \cdot \overline{\operatorname{diag}(H)} \cdot I,
\qquad \lambda = \text{damp\_pct}
$$

The sequential update rule needs $H_{\text{damped}}^{-1}$ specifically
through its **upper-triangular Cholesky factor** $U$, i.e. the matrix
satisfying

$$
H_{\text{damped}}^{-1} = U^\top U, \qquad U \text{ upper triangular}
$$

(a standard *lower* Cholesky factor $L$ of $H_{\text{damped}}^{-1}$
already gives this directly: $U = L^\top$, since $L L^\top =
H_{\text{damped}}^{-1}$ means $(L^\top)^\top(L^\top) = LL^\top$ too, and
$L^\top$ is upper triangular).

## Task

Implement `damped_inv_hessian_cholesky`:

```python
def damped_inv_hessian_cholesky(X: np.ndarray, damp_pct: float) -> dict:
    ...
```

- `X`: `(n_cal, d_in)` calibration activations.
- `damp_pct`: float in `(0, 1)`, e.g. `0.01`.
- Compute `H = 2 X^T X`, damp it as above, invert the damped matrix, and
  take the upper Cholesky factor of the inverse.

Return a dict:
- `"H"`: the **damped** Hessian, `(d_in, d_in)`.
- `"Hinv"`: its inverse, `(d_in, d_in)`.
- `"U"`: the upper-triangular Cholesky factor of `Hinv`, `(d_in, d_in)`.

## Example

```python
import numpy as np

X = np.random.default_rng(0).standard_normal((80, 6))
out = damped_inv_hessian_cholesky(X, damp_pct=0.01)

# sanity checks any correct implementation satisfies:
assert np.allclose(out["H"] @ out["Hinv"], np.eye(6), atol=1e-6)
assert np.allclose(out["U"], np.triu(out["U"]))
assert np.allclose(out["U"].T @ out["U"], out["Hinv"], atol=1e-6)
```

## What the gate checks

The grader loads a committed `X.npy` fixture (calibration activations
with a correlated mixing matrix, so `X^T X` is strongly non-diagonal —
the regime GPTQ's Hessian-based correction actually matters in), graded
with `damp_pct = 0.01`, plus several additional seeded synthetic
`(X, damp_pct)` pairs, and computes `H`, `Hinv`, `U` independently with
plain NumPy linear algebra — never calling your function, never
hardcoding an expected matrix.

`rel_err` is `scorers.rel_err` applied to the concatenation of all three
returned matrices (flattened) against the oracle's, taking the worst
case across all cases, and must be `<= 1e-6`. Damping before vs. after
computing the mean diagonal, damping only some entries, inverting the
undamped `H`, or returning the lower instead of the upper Cholesky
factor will all show up as a large mismatch in at least one of the three
matrices.
