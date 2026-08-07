## Context

The scaled dot‑product attention (SDPA) mechanism is a core component of transformer models.  
Given query, key and value matrices $Q \in \mathbb{R}^{b\times d_k}$, $K \in \mathbb{R}^{d_k\times d_k}$ and $V \in \mathbb{R}^{d_k\times d_v}$, the attention output is

$$
\operatorname{Attention}(Q,K,V)
  = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

The softmax is applied row‑wise to the $b\times d_k$ logits matrix.  
This operation can be implemented efficiently with Python broadcasting and matrix multiplication, without explicit Python loops.

## Task

Implement a function `sdpa(Q: list[float], K: list[float], V: list[float]) -> list[float] that returns the attention output as described above. The implementation must use only vectorised Python operations; no Python `for` loops are allowed. The result should have shape `(b, d_v)` and be of type `float32` if the inputs are `float32`, otherwise preserve the input dtype.

```python
def sdpa(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    ...
```

## Example

```python

Q = [[1., 0.], [0., 1.]]          # shape (2, 2)
K = [[1., 0.], [0., 1.]]          # shape (2, 2)
V = [[1., 2.], [3., 4.]]          # shape (2, 2)

out = sdpa(Q, K, V)
print(out)  # [[1.6604769013466862, 2.6604769013466862], [2.3395230986533138, 3.3395230986533138]]
```

The example uses identity matrices for $Q$ and $K$, so the softmax produces a one‑hot distribution that selects each row of $V$ unchanged.

## What the gate checks

Two aspects are verified:

* **Numerical correctness** – The maximum absolute difference between your output and a Python reference implementation must be at most $10^{-6}$.
* **Pure vectorisation** – Your code should not contain any Python `for` loops; only Python operations are allowed.

The gate uses the scorer `max_abs_err` from `arena.scorers`. A failing implementation will produce a larger error or raise an exception, causing the gate to fail.
