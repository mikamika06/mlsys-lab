## Context

In standard multi‑head attention each query head attends to all key/value heads. In grouped‑query attention (GQA) a single key/value head is shared by a group of $g$ query heads, reducing computation and memory. For a batch of queries $Q \in \mathbb{R}^{n_q\times d}$, keys $K \in \mathbb{R}^{n_{kv}\times d}$ and values $V \in \mathbb{R}^{n_{kv}\times d}$, the attention score for query $i$ that is assigned to key/value head $j = \lfloor i/g\rfloor$ is

$$
s_i = Q_i^\top K_j,
$$

and the output vector is simply $O_i = s_i\,V_j$. The assignment of queries to KV heads is deterministic and depends only on the grouping factor $g$.

## Task

Implement `gqa_attention(Q, K, V, g)` that returns a list `O` of shape `(n_q, d)`. Use vectorized Python operations only; no explicit Python loops. The function must work for arbitrary positive integers `n_q`, `n_kv`, `d`, and grouping factor `g` such that $n_q = n_{kv}\times g$.

```python
def gqa_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], g: int) -> list[list[float]]:
    ...
```

The output should be of dtype `float64`.

## Example

```python
Q = [[1., 0.], [0., 1.], [1., 1.]]
K = [[2., 3.], [4., 5.]]   # n_kv=2, g=2 -> n_q=4 but we use 3 for demo
V = [[7., 8.], [9.,10.]]
O = gqa_attention(Q, K, V, g=2)
print(O)
```

Output (rounded):

```
[[14.0, 16.0], [21.0, 24.0], [81.0, 90.0]]
```

## What the gate checks

The grader computes a reference implementation with Python and compares your output to it using the scorer `max_abs_err`. The maximum absolute difference must be at most $10^{-5}$.
