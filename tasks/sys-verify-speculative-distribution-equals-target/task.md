## Context

Speculative decoding accelerates autoregressive generation by using a smaller draft model to propose tokens and a larger target model to verify them.

Let $q(x)$ be the draft distribution and $p(x)$ be the target distribution over a vocabulary. A proposed token $x$ from the draft model is accepted with probability

$$
\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right).
$$

If the proposal is rejected, speculative decoding samples from the residual distribution

$$
r(x) = \frac{\max(p(x)-q(x), 0)}{\sum_j \max(p(j)-q(j),0)} .
$$

The accepted-or-resampled output distribution should match the target distribution $p$. This task verifies the distributional property by running many sampling steps and comparing the empirical output distribution with the target.

## Task

Implement `speculative_distribution(draft_probs, target_probs, steps, seed)`:

```python
def speculative_distribution(
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
    steps: int,
    seed: int,
) -> np.ndarray:
    ...
```

The inputs are one-dimensional probability arrays of equal length. They contain non-negative values that sum to $1$. The function must simulate speculative decoding for `steps` generated tokens and return the empirical token distribution as a one-dimensional `float64` NumPy array.

For each step:

1. Sample a draft token from $q$.
2. Accept it with probability $\min(1, p(x)/q(x))$.
3. Otherwise sample from the normalized residual distribution.

Use the provided seed with NumPy's random generator so that the result is deterministic.

## Example

```python
import numpy as np

draft = np.array([0.70, 0.20, 0.10])
target = np.array([0.40, 0.35, 0.25])

out = speculative_distribution(draft, target, 100000, 7)

# out is close to:
# [0.40, 0.35, 0.25]
```

## What the gate checks

The gate independently implements the speculative decoding sampler using NumPy as the oracle. It runs several seeded cases and measures

$$
\mathrm{KL}(p \Vert \hat{p}) =
\sum_i p_i \log\frac{p_i}{\hat{p}_i},
$$

where $p$ is the target distribution and $\hat{p}$ is the empirical distribution returned by the candidate.

The reported `mean_kl` is the average KL value over the cases. The value must be less than $5 \times 10^{-3}$.
