## Context

Computing the mean and variance of a sequence in a **single pass** is essential for
GPU LayerNorm and RMSNorm kernels, where re-reading data costs extra global-memory
bandwidth.

**Welford's online algorithm** maintains three accumulators $(n, \mu, M_2)$ and
updates them for each new value $x_k$:

$$n \leftarrow n + 1, \quad \delta \leftarrow x_k - \mu, \quad \mu \leftarrow \mu + \frac{\delta}{n}, \quad \delta_2 \leftarrow x_k - \mu, \quad M_2 \leftarrow M_2 + \delta \cdot \delta_2$$

After all $N$ elements:

$$\text{mean} = \mu, \qquad \text{variance} = \frac{M_2}{N}$$

This is numerically more stable than the naive two-pass formula for wide-dynamic-range
inputs.

## Task

Implement, as real CUDA-C:

```cpp
__global__ void welford_kernel(const float* x, float* out, int n);
```

Compute the **population mean and variance** of `x[0..n)` using Welford's
one-pass algorithm, **in a single thread** (the lane with
`blockIdx.x * blockDim.x + threadIdx.x == 0` — every other lane should
`return;` immediately and do no work). Write:

- `out[0]` $\leftarrow$ mean
- `out[1]` $\leftarrow$ variance ($M_2 / n$)

## Example

For `x = [1.0, 2.0, 3.0, 4.0, 5.0]` (`n = 5`): `out[0] = 3.0` (mean),
`out[1] = 2.0` (population variance).

## What the gate checks

The grader launches your kernel (`grid=1, block=32`, one warp) on a
wide-dynamic-range fixture (100 values mixing magnitudes around $10^{-4}$
and $10^{4}$), reads back `out[0]`/`out[1]`, and computes the reference
mean and variance independently with NumPy's two-pass algorithm. It checks
$\mathrm{max\_abs\_err} \le 10^{-6}$ between
$[\mu_{\text{yours}}, \sigma^2_{\text{yours}}]$ and
$[\mu_{\text{ref}}, \sigma^2_{\text{ref}}]$. The compiled reference lands
around $3.6\times10^{-12}$ — the wide dynamic range is exactly the case
where a naive single running sum (instead of Welford's incremental mean
update) would lose precision.
