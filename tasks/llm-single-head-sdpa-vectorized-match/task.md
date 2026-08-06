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
The implementation must use only Python vectorised operations; no explicit Python loops are allowed.  
All computations should be performed in double precision (`float64`).

## Example

```python
Q = [[1., 0.], [0., 1.]]
K = Q.copy()
V = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
out = sdpa_single_head(Q, K, V)
print(out)  # [[0.6697615493266569, 0.3302384506733431], [0.3302384506733431, 0.6697615493266569]]
```

## What the gate checks

The grader computes a reference implementation with Python and compares your output to it using the metric `max_abs_err`.  
Your solution must satisfy

$$
\max_{i,j} |\, \text{your}(i,j) - \text{reference}(i,j)\,| \le 10^{-6}.
$$

The function is also required to run in a single vectorised call; any Python loop will cause the gate to fail.
