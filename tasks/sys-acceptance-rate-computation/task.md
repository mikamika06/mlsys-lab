## Context

In speculative decoding a draft distribution $q$ is generated for each token position and compared to the target distribution $p$. The probability that the draft will be accepted at a given position equals the overlap of the two distributions, i.e.

$$\alpha_i = \sum_{v} \min(p_{i,v}, q_{i,v})\,.$$

The overall acceptance rate is the mean of $\alpha_i$ over all positions. For this task we only need to compute the per‑position values $\alpha_i$.

## Task

Implement `acceptance_rate(target, draft)` that takes two 2‑D NumPy arrays of shape $(n,\text{vocab})$ containing probability distributions (rows sum to one) and returns a 1‑D array of length $n$ with the expected acceptance rate at each position. The result must be of dtype `float64`.

## Example

```python
import numpy as np
target = np.array([[0.7, 0.3],
                   [0.4, 0.6]])
draft  = np.array([[0.5, 0.5],
                   [0.1, 0.9]])

rates = acceptance_rate(target, draft)
# array([0.8, 0.5])
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your output using the `rel_err` scorer from `arena.scorers`. Your solution must achieve a relative error $\le 10^{-9}$.
