## Context

Mistral replaces the dense causal attention of a vanilla transformer with
**sliding-window attention**. Instead of letting every query look back at the
whole past, each query position $i$ may attend only to the $w$ most recent key
positions — the half-open window $(i-w,\; i]$. Concretely, the key position $j$
is visible to query $i$ iff

$$
i - w < j \le i
\qquad\Longleftrightarrow\qquad
j \le i \;\text{ and }\; i - j < w .
$$

This is causal (a token never sees the future, $j \le i$) **and** bounded (a
token never looks back more than $w$ steps). Positions outside the window are
removed *before* the softmax by setting their score to $-\infty$, so they carry
exactly zero probability mass.

Given a single head with query, key and value matrices
$Q, K, V \in \mathbb{R}^{n \times d}$, the windowed attention output is

$$
\operatorname{Att}_w(Q,K,V)
  = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}} + M_w\right) V ,
\qquad
(M_w)_{ij} =
\begin{cases}
0 & i - w < j \le i,\\[2pt]
-\infty & \text{otherwise.}
\end{cases}
$$

The softmax is taken row-wise over the $n \times n$ score matrix. Because $j=i$
is always inside the window (for $w \ge 1$), every row has at least one visible
key, so the softmax is always well defined.

## Task

Implement `sliding_window_attention(Q, K, V, w)`:

```python
def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, w: int) -> np.ndarray:
    ...
```

- `Q`, `K`, `V` are 2-D NumPy arrays of shape $(n, d)$ (one attention head).
- `w` is a positive integer window width — the number of key positions each
  query may attend to (itself plus the $w-1$ preceding tokens).
- Return the $(n, d)$ attention output defined above. Apply the window mask
  **before** the softmax (masked scores become $-\infty$), scale the scores by
  $1/\sqrt{d}$, and compute everything in double precision (`float64`).
- If $w \ge n$ the window never clips and the result equals ordinary causal
  attention.

## Example

```python
import numpy as np
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
K = Q.copy()
V = np.array([[10., 0.], [0., 20.], [5., 5.]])

out = sliding_window_attention(Q, K, V, w=1)
# w = 1 -> each query attends only to itself, so the output equals V:
# [[10.  0.]
#  [ 0. 20.]
#  [ 5.  5.]]
```

## What the gate checks

The grader builds its own reference with NumPy — it constructs the boolean
window mask $i - w < j \le i$, applies it to $QK^\top/\sqrt{d}$ before a stable
row-wise softmax, and multiplies by $V$ — over several sizes $(n, d)$ and window
widths $w$ (including $w=1$, a mid-range window, and $w \ge n$). It compares your
output to that reference with the metric `max_abs_err`. Your solution passes iff

$$
\max_{i,j} \bigl|\, \text{your}(i,j) - \text{reference}(i,j) \,\bigr| \le 10^{-6}.
$$
