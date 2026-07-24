## Context

For a cache-blocked $N \times N$ matmul, the classic memory-traffic result
is that DRAM traffic scales as

$$\text{traffic} = O\!\left(\frac{N^3}{\sqrt{C}}\right)$$

where $C$ is the cache capacity: the optimal block size is
$B \approx \sqrt{C/3}$ (three $B \times B$ float tiles — one from each of
$A$, $B$, $C$ — must fit in the cache simultaneously), and each of the
$O(N^3/B^2)$ block-triples does $O(B^2)$ work while reloading $O(B^2)$
bytes, so total traffic is $O(N^3/B) = O(N^3/\sqrt{C})$. Doubling the
cache capacity should cut DRAM traffic by roughly $\sqrt{2}$, not by
$2\times$ — a very different scaling than compute, which does not care
about $C$ at all.

The standard way to confirm a power law $y = a \cdot x^b$ empirically is
to take logs, turning it into a straight line
$\ln(y) = \ln(a) + b \cdot \ln(x)$, and fit $b$ with ordinary least
squares.

## Task

Implement

```cpp
double fit_scaling_exponent(const double* x, const double* y, int n);
```

Given `n` positive samples `(x[i], y[i])`, fit the exponent `b` of the
power law `y = a * x^b` by ordinary least squares on the log-log
transformed data:

$$b = \frac{\sum_i (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_i (X_i - \bar{X})^2}, \qquad X_i = \ln x_i,\ Y_i = \ln y_i$$

## Example

The driver runs a real cache-blocked matmul ($N = 128$) against a
deterministic direct-mapped cache model at 5 capacities
($1024, 2048, 4096, 8192, 16384$ bytes) and measures the actual DRAM
traffic (miss count $\times$ line size) at each one — no numbers are
hardcoded, they come from really running the blocked algorithm through
the cache model. `fit_scaling_exponent` is then called on those 5
`(capacity, traffic)` points. The fitted exponent comes out close to
$-0.5$ (traffic shrinks roughly with $1/\sqrt{C}$, as the theory
predicts), though boundary and line-quantization effects on a small,
real sweep keep it from landing exactly on $-0.5$.

## What the gate checks

`max_abs_err` on the printed traffic samples and the fitted exponent:
the driver's data-generation step is identical for every candidate, so
only the regression itself is being graded. Swapping $x$ and $y$,
forgetting the logs (fitting a linear rather than a power law), or using
the wrong mean all change the slope enough to fail; a starter that
returns `0.0` fails outright since the real fitted exponent is nowhere
near zero.
