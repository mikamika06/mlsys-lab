## Context

A per-tensor quantizer sets its single scale from the largest-magnitude
element in the tensor. If a linear layer's activations have a few outlier
channels 50-200x larger than the rest, that shared scale is set almost
entirely by the outliers, and every ordinary channel gets crushed into a
handful of quantization levels.

QuaRot / SpinQuant fix this with a rotation. Let $H$ be an orthogonal
matrix ($H^{\top} H = I$) — a normalized Sylvester-Hadamard matrix is a
convenient, cheap-to-apply choice. Fold it into both operands of a matmul:

$$
X' = X H^{\top}, \qquad W' = H W.
$$

The product is exactly unchanged:

$$
X' W' = X H^{\top} H W = X (H^{\top}H) W = X W,
$$

because $H^{\top} H = I$. But $H$ mixes every input channel into every
rotated channel — a single huge outlier channel's energy gets spread across
all $d$ rotated channels instead of sitting in one place, so the *rotated*
tensor's per-tensor quantization scale is far less outlier-dominated than
the original's.

## Task

Implement `rotate_and_quantize_matmul(X, W)`.

- `X`: activations, shape `(n, d)`, with `d` a power of two.
- `W`: weights, shape `(d, m)`.

Steps:

1. Build the normalized Sylvester-Hadamard matrix $H$ of size `d` (the
   recursive construction $H_1 = [1]$, $H_{2k} =
   \frac{1}{\sqrt{2}}\begin{psmallmatrix}H_k & H_k \\ H_k &
   -H_k\end{psmallmatrix}$, normalized so $H^{\top}H = I$).
2. Compute `X' = X @ H.T`, `W' = H @ W`, and `out_rotated = X' @ W'`.
3. Symmetric per-tensor int4 round-trip a tensor `t`: `qmax = 7`,
   `scale = max(|t|) / qmax` (use `scale = 1.0` if `t` is all-zero),
   `code = clip(round(t / scale), -qmax, qmax)`, `dequant = code * scale`.
4. `mse_unrotated = mean((X @ W - Xq @ Wq) ** 2)`, where `Xq`, `Wq` are the
   int4 round-trip of the *unrotated* `X`, `W`.
5. `mse_rotated = mean((X @ W - Xq' @ Wq') ** 2)`, where `Xq'`, `Wq'` are
   the int4 round-trip of the *rotated* `X'`, `W'`.

Return `(out_rotated, mse_unrotated, mse_rotated)`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 32))
X[:, 5] *= 80.0          # one big outlier channel
W = rng.normal(size=(32, 8)) * 0.3

out_rotated, mse_unrotated, mse_rotated = rotate_and_quantize_matmul(X, W)
assert np.allclose(out_rotated, X @ W, rtol=1e-8, atol=1e-8)
assert mse_rotated < mse_unrotated
```

## What the gate checks

The gate rebuilds the same rotation and int4 round-trip with an independent
NumPy oracle across several `(X, W)` pairs, each with one or two extreme
outlier activation channels and `d` in `{32, 64, 128}`:

- `invariance_rel_err`: the relative error between your `out_rotated` and
  the oracle's `X @ W` must be at most `1e-5` (the rotation must actually
  be invariant, not approximately so).
- `mse_rel_err`: your reported `mse_unrotated` and `mse_rotated` must match
  the oracle's to a relative tolerance of `1e-6`.
- `rotation_helps`: for every test case, your own `mse_rotated` must be
  strictly smaller than your own `mse_unrotated` — quantizing after
  rotation must actually reduce error, not just produce numbers that
  happen to be close to the oracle's.

A solution that applies `H` to only one of `X`/`W` (breaking invariance),
or that quantizes `X'`/`W'` with a *per-channel* scheme instead of a single
shared per-tensor scale (which would hide the outlier-spreading effect the
gate is testing for), will fail one of these three gates.
