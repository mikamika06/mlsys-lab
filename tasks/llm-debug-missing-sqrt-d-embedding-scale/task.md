## Context

In transformer models the embedding matrix $E \in \mathbb{R}^{V\times d}$ is usually divided by $\sqrt{d}$ before being used in the attention or classification head. This scaling keeps the variance of dot products stable as $d$ grows. If omitted, logits become too large and training diverges.

Mathematically we want

$$\tilde E = \frac{1}{\sqrt{d}}\,E.$$

## Task

Implement `normalize_embeddings(embeddings: np.ndarray) -> np.ndarray` that returns the scaled matrix. The input is a 2‑D NumPy array of shape $(V,d)$ and may contain any numeric dtype; output must be float64.

## Example

```python
import numpy as np
E = np.array([[1, 2], [3, 4]], dtype=np.float32)
S = normalize_embeddings(E)
# S == [[0.5, 1.0],
#       [1.5, 2.0]]
```

## What the gate checks

The grader computes a reference scaling with NumPy and compares the maximum absolute difference between your output and the reference using `arena.scorers.max_abs_err`. The error must be at most $10^{-6}$.
