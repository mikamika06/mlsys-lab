## Context

Scaled dot‑product attention is the core of transformer models.  
For a single head we have query matrix $Q \in \mathbb{R}^{n\times d}$, key matrix $K \in \mathbb{R}^{n\times d}$ and value matrix $V \in \mathbb{R}^{n\times d}$.  
The attention output is

$$
\operatorname{Att}(Q,K,V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right)\! V .
$$

The softmax is applied row‑wise to the $n\times n$ score matrix.

## Task

Implement a function `sdpa_single_head(Q, K, V)` that returns the attention output as defined above.  
The implementation must use only NumPy vectorised operations; no explicit Python loops are allowed.  
All computations should be performed in double precision (`float64`).

## Example

```python
import numpy as np
Q = np.array([[1., 0.], [0., 1.]])
K = Q.copy()
V = np.eye(2)
out = sdpa_single_head(Q, K, V)
print(out)
# [[0.5 0.5]
#  [0.5 0.5]]
```

## What the gate checks

The grader computes a reference implementation with NumPy and compares your output to it using the metric `max_abs_err`.  
Your solution must satisfy

$$
\max_{i,j} |\, \text{your}(i,j) - \text{reference}(i,j)\,| \le 10^{-6}.
$$

The function is also required to run in a single vectorised call; any Python loop will cause the gate to fail.
