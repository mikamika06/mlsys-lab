## Context

In transformer models the embedding matrix $E \in \mathbb{R}^{V\times d}$ is usually divided by $\sqrt{d}$ before being used in the attention or classification head. This scaling keeps the variance of dot products stable as $d$ grows. If omitted, logits become too large and training diverges.

Mathematically we want

$$\tilde E = \frac{1}{\sqrt{d}}\,E.$$

## Task

Implement `normalize_embeddings(embeddings: list[float]) -> list[float] that returns the scaled matrix. The input is a 2‑D list of shape $(V,d)$ and may contain any numeric dtype; output must be float64.

## Example

```python
E = [[1, 2], [3, 4]]
S = normalize_embeddings(E)
# S == [[0.5, 1.0],
#       [1.5, 2.0]]
```

## What the gate checks

The grader computes a reference scaling with Python and compares the maximum absolute difference between your output and the reference using `arena.scorers.max_abs_err`. The error must be at most $10^{-6}$.
