## Context

LayerNorm's forward pass normalizes each row of `x` independently over
its `D` features, then applies a learned scale `gamma` and shift `beta`:

$$
\hat{x}_i = \frac{x_i - \mu}{\sigma}, \qquad y_i = \gamma_i \hat{x}_i + \beta_i,
\qquad \mu = \frac{1}{D}\sum_i x_i,\ \ \sigma = \sqrt{\frac{1}{D}\sum_i (x_i-\mu)^2 + \epsilon}
$$

Given the upstream gradient $dy$, the backward pass needs three things.
`dbeta` and `dgamma` are per-feature sums **across every row** (since
`beta`/`gamma` are shared by the whole batch):

$$
d\beta_j = \sum_{\text{rows}} dy_j, \qquad
d\gamma_j = \sum_{\text{rows}} dy_j \, \hat{x}_j
$$

`dx` is per-element but depends on **the whole row** it came from,
because $\mu$ and $\sigma$ were both computed from every element of
that row. Writing $g_i = dy_i \cdot \gamma_i$ (the gradient w.r.t.
$\hat{x}_i$):

$$
dx_i = \frac{1}{\sigma}\Big(g_i - \overline{g} - \hat{x}_i \, \overline{g\hat{x}}\Big)
$$

where $\overline{g}$ and $\overline{g\hat{x}}$ are means over the row's
$D$ features. This subset has no atomics and no cross-block reduction
primitive, so `dgamma`/`dbeta`'s across-row sum can't be built by
having many threads increment a shared accumulator; every thread must
compute whatever sum it needs entirely on its own.

## Task

Implement:

```cuda
__global__ void layernorm_backward(float* dx, float* dgamma, float* dbeta, const float* dy, const float* x, const float* gamma, int B, int D);
```

All of `dx`, `dy`, `x` are flattened $B \times D$ (row-major); `gamma`,
`dgamma`, `dbeta` have $D$ entries. Launch with one thread per element,
`blockDim.x = B*D` (this task fixes `B=4, D=8`). Thread
`tid = threadIdx.x` owns `row = tid/D`, `col = tid%D`:

1. Compute row `row`'s mean and biased variance over its `D` features
   (`eps = 1e-5`), then `std = sqrt(var + eps)`.
2. Compute $\overline{g}$ and $\overline{g\hat{x}}$ over row `row`'s
   `D` features (`g_j = dy[row,j] * gamma[j]`).
3. Write `dx[row,col]` using the formula above.
4. Recompute the SAME mean/var/std for every row `r` in `[0, B)`, and
   accumulate `dgamma[col] += dy[r,col] * xhat[r,col]`,
   `dbeta[col] += dy[r,col]`, writing the final sums to
   `dgamma[col]`/`dbeta[col]`. (Every thread that shares column `col`
   computes and writes the identical value, so multiple threads
   targeting the same output slot is safe.)

## Example

For a single-row, `D=2` case with `x=[1,3]`, `gamma=[1,1]`, `dy=[1,-1]`:
`mean=2`, `var=1`, `std≈1.0000`, `xhat≈[-1,1]`. `g=[1,-1]`,
`mean(g)=0`, `mean(g*xhat)=(1*-1 + -1*1)/2 = -1`. `dx_0 = (1 - 0 -
(-1)*(-1))/1 = 0`, `dx_1 = (-1 - 0 - 1*(-1))/1 = 0` (a single-row
batch's `dx` sums to zero exactly, since shifting every `dx_i` by any
constant doesn't change the normalized loss -- a useful sanity check).

## What the gate checks

`check.py` runs the kernel over a fixed `4x8` `x`/`dy` and length-8
`gamma`, and checks `max_abs_err <= 1e-6` against `numpy`'s
closed-form analytic gradient (computed independently via the same
formulas, not via autograd, but equivalent) across `dx`, `dgamma`, and
`dbeta` together. Missing the $\overline{g}$ or $\overline{g\hat{x}}$
correction terms in `dx` (i.e. computing it as if $\mu,\sigma$ were
constants, `dx_i = g_i / \sigma`) gives the wrong gradient for every
row; omitting rows from the `dgamma`/`dbeta` sums under-counts them.
