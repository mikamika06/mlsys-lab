## Context

In language‑model decoding we often restrict the set of candidate tokens to a small subset before applying softmax or other operations. One common strategy is **top‑k filtering**: given a vector of logits $\mathbf{z}\in\mathbb{R}^n$, we keep only the $k$ largest entries and mask all others, typically by setting them to $-\infty$. The kept set is then used for subsequent sampling or deterministic decoding. This operation is purely deterministic once $k$ is fixed.

The function you will implement receives a 1‑D NumPy array of logits and an integer $k$, and must return two things:

* A boolean mask $\mathbf{m}\in\{0,1\}^n$ where $m_i=1$ iff the $i$‑th logit is among the top $k$.
* The filtered logits $\hat{\mathbf{z}}$ of the same shape as $\mathbf{z}$, with all discarded entries replaced by $-\infty$.

The mask and the filtered logits must be NumPy arrays of type `bool` and `float64`, respectively. No Python loops are allowed; use vectorised NumPy operations only.

## Task

Implement the function:

```python
def top_k_filter(logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

The input array will always be one‑dimensional and contain at least $k$ elements. The output must satisfy the contract described above.

## Example

```python
import numpy as np
logits = np.array([0.5, 3.2, 1.7, 4.0])
mask, filtered = top_k_filter(logits, k=2)
print(mask)      # [False  True False  True]
print(filtered)  # [-inf  3.2 -inf  4.0]
```

Here the two largest logits are $4.0$ and $3.2$, so only those positions remain.

## What the gate checks

The grader computes a reference solution with NumPy’s `argsort` and compares the returned mask to the exact set of indices that should be kept. The filtered logits are also checked for correct masking. If either array differs, the candidate fails the **exact_match** gate. No other metrics are evaluated.
