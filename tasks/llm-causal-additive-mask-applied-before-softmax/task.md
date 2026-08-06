## Context

In transformer models, attention scores are computed as dot products between query and key vectors. To enforce causality in autoregressive generation, a *causal mask* is applied so that each position can attend only to itself and previous positions. The mask is usually an additive matrix with $-\infty$ for forbidden entries, which after the softmax turns those probabilities into zero.

The softmax over a vector $z \in \mathbb{R}^n$ is

$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)} .
$$

When an entry of $z$ is set to $-\infty$, $\exp(-\infty)=0$ and the corresponding probability becomes exactly zero.

## Task

Implement `causal_masked_softmax(scores)`:

```python
def causal_masked_softmax(scores: list[list[float]]) -> list[list[float]]:
    ...
```

`scores` is a 2‑D list of shape $(L, L)$ containing raw attention logits. The function must return an array of the same shape where each row has been softmaxed after applying a lower‑triangular causal mask (including the diagonal). The output should be of type `float64`.

## Example

```python
scores = [[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]]

masked = causal_masked_softmax(scores)
print(masked)  # [[1.0, 0.0, 0.0], [0.2689414213699951, 0.7310585786300048, 0.0], [0.09003057317038046, 0.24472847105479767, 0.665240955774822]]
```

The first row is softmaxed normally; the second and third rows have all future positions masked to $-\infty$ before the softmax, so only the diagonal entry survives.

## What the gate checks

The grader computes a reference implementation using list comprehensions and `math.exp`. It then evaluates the maximum absolute error between your output and the reference. Your solution must achieve

$$
\max_{i,j} |\, \hat{y}_{ij} - y_{ij}\,| \le 10^{-6}.
$$

The gate also verifies that the returned array has dtype `float64` and the same shape as the input.
