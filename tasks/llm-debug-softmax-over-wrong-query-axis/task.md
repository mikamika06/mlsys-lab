## Context

Scaled dot‑product attention (SDPA) is a core component of transformer models.  
Given queries $Q \in \mathbb{R}^{B\times N_q\times d_k}$, keys $K \in \mathbb{R}^{B\times N_k\times d_k}$ and values $V \in \mathbb{R}^{B\times N_k\times d_v}$, the attention output is

$$
\operatorname{SDPA}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)\!V .
$$

The softmax must be applied over the **key** dimension $N_k$ (the last axis of the score matrix).  A common bug is to apply it over the query dimension instead, which swaps the roles of queries and keys and produces incorrect results.

## Task

Implement `sdpa(query, key, value, scale=None)`:

```python
def sdpa(query: list[list[list[float]]],
         key: list[list[list[float]]],
         value: list[list[list[float]]],
         scale: float | None = None) -> list[list[list[float]]]:
    ...
```

* `query`, `key` and `value` are 3‑D list with shapes  
  $(B, N_q, d_k)$, $(B, N_k, d_k)$ and $(B, N_k, d_v)$ respectively.
* If `scale` is `None`, use the default scaling factor $1/\sqrt{d_k}$.
* Return a list of shape $(B, N_q, d_v)$ containing the attention output.

The implementation must be fully vectorised; no explicit Python loops over batch or sequence indices are allowed.  The function should raise a `ValueError` if the input shapes are incompatible.

## Example

```python

Q = [[[1., 0.], [0., 1.]]]          # shape (1,2,2)
K = [[[1., 0.], [0., 1.], [1., 1.]]]# shape (1,3,2)
V = [[[1., 0.], [0., 1.], [1., 1.]]]# shape (1,3,2)

out = sdpa(Q, K, V)
print(out.shape)   # (1, 2, 2)
print(out[0])
```

The output should be a $(2\times 2)$ matrix of attention values.

## What the gate checks

Two metrics are evaluated:

* **max_abs_err** – the maximum absolute difference between your result and a Python reference implementation.  
  The gate requires `max_abs_err <= 1e-6`.

The grader uses a real Python oracle to compute the reference, so any hard‑coded expected values will not pass.
