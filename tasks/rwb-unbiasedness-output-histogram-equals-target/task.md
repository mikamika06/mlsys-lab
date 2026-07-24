## Context

Speculative sampling accelerates autoregressive generation by using a proposal
distribution $q$ and correcting it so that the final samples follow the target
distribution $p$.

For a proposed token $y$ drawn from $q$, the proposal is accepted with
probability

$$
\alpha(y) = \min\left(1, \frac{p(y)}{q(y)}\right).
$$

If the proposal is rejected, speculative sampling draws from the residual
distribution

$$
r(i) =
\frac{\max(p(i)-q(i),0)}
{\sum_j \max(p(j)-q(j),0)}.
$$

The accepted and residual paths together preserve the target distribution:

$$
P(\text{output}=i)=p(i).
$$

The output distribution can be measured by an empirical histogram. For $N$
independent samples, the expected sampling error decreases approximately as
$O(1/\sqrt{N})$.

## Task

Implement `speculative_histogram(p, q, seed, n_samples)`:

```python
def speculative_histogram(
    p: np.ndarray,
    q: np.ndarray,
    seed: int,
    n_samples: int
) -> np.ndarray:
    ...
```

The inputs are one-dimensional NumPy arrays containing valid probability
distributions over the same vocabulary. Use the provided random seed to make
the result deterministic. Run the speculative accept/resample algorithm
`n_samples` times and return a one-dimensional array containing the empirical
token frequencies.

The returned histogram must contain probabilities, not integer counts. The
output shape must equal `p.shape` and the values should sum to $1$.

## Example

```python
import numpy as np

p = np.array([0.5, 0.3, 0.2])
q = np.array([0.4, 0.4, 0.2])

hist = speculative_histogram(p, q, seed=7, n_samples=10000)

# hist is close to:
# [0.5, 0.3, 0.2]
```

## What the gate checks

The gate runs the implementation on several target and proposal distributions
with fixed seeds and sample counts.

The checker computes the oracle distribution directly from the target
probability vector $p$. It measures the relative error

$$
\mathrm{rel\_err}
=
\frac{\lVert h-p\rVert_2}
{\lVert p\rVert_2 + 10^{-12}},
$$

where $h$ is the returned histogram.

The implementation passes when the measured error stays below the Monte Carlo
tolerance. Returning samples directly from $q$ fails because the proposal
distribution is not generally the target distribution.
