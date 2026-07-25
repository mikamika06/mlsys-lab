## Context

LayerNorm normalizes each row of activations independently — unlike
BatchNorm, it never mixes statistics across different rows (samples),
which is exactly why it works for variable batch sizes and sequence
models. For a row $x \in \mathbb{R}^D$:

$$
\mu = \frac{1}{D}\sum_d x_d, \qquad
\sigma^2 = \frac{1}{D}\sum_d x_d^2 - \mu^2, \qquad
y_d = \frac{x_d - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_d + \beta_d
$$

$\gamma$ and $\beta$ are the learned per-feature scale and shift
(shared across every row); $\epsilon$ keeps the division well-defined
when a row's variance is tiny. Computing $\sigma^2$ from
$\mathbb{E}[x^2] - \mathbb{E}[x]^2$ lets the whole thing run in exactly
two sweeps over the row — one to accumulate both sums, one to write the
normalized output — instead of one pass for the mean and a second pass
to re-read $x$ for the variance.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void layernorm_forward(const float* x, const float* gamma, const float* beta,
                                   float* y, int rows, int D, float eps);
```

One thread per row (`row = blockIdx.x*blockDim.x + threadIdx.x`). For
`row < rows`:

1. Sweep `d` from `0` to `D-1`, accumulating `sum += x[row*D+d]` and
   `sumsq += x[row*D+d]^2`.
2. `mean = sum / D`, `var = sumsq/D - mean*mean`,
   `invstd = 1 / sqrt(var + eps)`.
3. Sweep `d` again, writing
   `y[row*D+d] = (x[row*D+d] - mean) * invstd * gamma[d] + beta[d]`.

## Example

Row `[1, 2, 3, 4]`, `gamma = beta` all `1`/`0` (identity affine),
`eps ≈ 0`: `mean = 2.5`, `var = mean(x^2) - mean^2 = 7.5 - 6.25 =
1.25`, `invstd = 1/sqrt(1.25) ≈ 0.894`. `y = [(1-2.5), (2-2.5),
(3-2.5), (4-2.5)] * 0.894 = [-1.342, -0.447, 0.447, 1.342]`.

## What the gate checks

The grader launches `layernorm_forward` on a fixed `16 x 8` input with
random `gamma`/`beta`, and compares the output against numpy's own
`(x - x.mean(1)) / sqrt(x.var(1) + eps) * gamma + beta`. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-5}
$$

Every row is normalized using **only that row's own** mean and
variance — no cross-row mixing, no reduction across threads needed,
since one thread owns one entire row start to finish.
