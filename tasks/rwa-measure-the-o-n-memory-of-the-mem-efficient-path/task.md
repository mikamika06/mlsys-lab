## Context

The softmax function is ubiquitous in machine learning for converting logits $z \in \mathbb{R}^K$ into a probability distribution:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{K}\exp(z_j)}.$$

A naïve implementation that first computes the full matrix of exponentials for an $N\times K$ logits tensor allocates an intermediate array of shape $(N,K)$, costing $\mathcal{O}(NK)$ memory.  In many production systems we avoid this by computing only two per‑row statistics:

* the maximum value $m_i = \max_j z_{ij}$,
* the sum of exponentials after shifting by that maximum
  $$s_i = \sum_{j=1}^{K}\exp(z_{ij} - m_i).$$

These two $\mathcal{O}(N)$ arrays are sufficient to reconstruct the softmax probabilities on demand and keep memory usage linear in $N$.

## Task

Implement `softmax_stats` that, given a 2‑D NumPy array of shape $(N,K)$ containing logits, returns a tuple `(row_max, row_sum_exp)` where:

* `row_max` is a 1‑D array of length $N$ with the per‑row maximum,
* `row_sum_exp` is a 1‑D array of length $N$ with the sum of exponentials after shifting by that maximum.

The implementation must **not** allocate an intermediate $(N,K)$ array; it should use only $\mathcal{O}(K)$ temporary memory per row. The result arrays must be of type `float64`.

```python
def softmax_stats(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Example

```python
import numpy as np
logits = np.array([[0., 1.], [2., -1.]])
row_max, row_sum_exp = softmax_stats(logits)
print(row_max)      # [1. 2.]
print(row_sum_exp)  # [exp(0)+exp(-1), exp(0)+exp(-3)]
```

## What the gate checks

* **exact_match** – The returned arrays must match a reference implementation that uses the full $(N,K)$ intermediate array.
* **size_ratio** – The ratio of the size (in bytes) of the naive $(N,K)$ exponential matrix to the combined size of the two $\mathcal{O}(N)$ statistic arrays must be at least $1.0$. This ensures the student’s solution does not allocate a large intermediate buffer.
