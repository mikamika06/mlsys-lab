## Context

The cross-entropy loss for a single sample with ground-truth class $t \in \{0,\dots,C-1\}$ and logits $\mathbf{x} \in \mathbb{R}^C$ is

$$ \ell(\mathbf{x}, t) = -\log\left( \frac{e^{x_t}}{\sum_{j=0}^{C-1} e^{x_j}} \right). $$

A naive implementation computes the softmax denominator $\sum_j e^{x_j}$ directly. When any $x_j$ is large (e.g.\ $10^3$), the exponent $e^{x_j}$ overflows to `inf` in IEEE-754 double precision, producing `NaN` after taking the logarithm.

The standard numerically stable formulation rewrites the loss by factoring out the maximum logit $m = \max_j x_j$:

$$
\ell(\mathbf{x}, t)
= -(x_t - m) + \log\sum_{j} e^{x_j - m}
$$

The shifted exponentials $e^{x_j - m}$ are at most $1$, so no overflow occurs. The $\log\!\sum e$ term equals $m + \log\!\sum_j e^{x_j - m}$, the classic log-sum-exp trick.

## Task

Implement the function

```python
def cross_entropy(logits: list[float], target: int) -> float:
```

where `logits` is a 1-D `float64` list of shape `(C,)` and `target` is an integer class index.

The provided starter code contains an **unstable** implementation that produces `NaN` for large logits. Replace it with a fused, numerically stable version.

You may only use Python (`import math`).

## Example

```python

logits = [1e3, 2e3, 3e3]
target = 1
loss = cross_entropy(logits, target)
print(loss)   # 1000.0  (finite)
```

The broken starter would return `NaN` because `[math.exp(x) for x in logits]` overflows.

## What the gate checks

The gate measures the relative L2 error

$$\mathrm{rel\_err} = \frac{\|\mathbf{L}_{\text{cand}} - \mathbf{L}_{\text{ref}}\|_2}{\|\mathbf{L}_{\text{ref}}\|_2 + 10^{-12}}$$

between vectors of losses from your implementation and a stable reference, evaluated on random test cases (moderate magnitudes) and extreme cases (logits up to $10^6$). The error must satisfy $\mathrm{rel\_err} \le 10^{-10}$.
