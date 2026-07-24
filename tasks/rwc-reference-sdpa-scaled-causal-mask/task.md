## Context

The scaled dot‑product attention mechanism is a core component of transformer models.  
Given query, key and value tensors $Q,K,V \in \mathbb{R}^{B\times N\times d}$ (batch size $B$, sequence length $N$), the unscaled scores are computed as
$$
S = Q K^\top,
$$
where each element $S_{b,i,j} = q_{b,i}\cdot k_{b,j}$ is a dot product between query at position $i$ and key at position $j$.  
To prevent large values from blowing up the softmax, the scores are divided by $\sqrt{d_k}$:
$$
\tilde S = \frac{S}{\sqrt{d_k}}.
$$

A *causal mask* forces each position to attend only to itself and earlier positions.  In matrix form this is a lower‑triangular mask $M$ with entries
$$
M_{i,j} =
\begin{cases}
0 & \text{if } j \le i,\\[4pt]
-\infty & \text{otherwise},
\end{cases}
$$
which is added to the scaled scores before softmax.  The attention weights are then
$$
A = \operatorname{softmax}(\tilde S + M),
$$
and the output of the layer is $O = A V$.

The implementation must be fully vectorised using NumPy and produce a `float64` array of shape $(B,N,d_v)$.

## Task

Implement the function

```python
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    *,
    causal: bool = False
) -> np.ndarray:
    ...
```

* `Q`, `K` and `V` are 3‑D NumPy arrays of shape `(B, N, d_k)` or `(B, N, d_v)` respectively.
* The function must compute the scaled dot‑product attention described above.
* If `causal=True`, a causal mask must be applied; otherwise no masking is performed.
* Return type must be `np.ndarray` with dtype `float64`.

## Example

```python
import numpy as np

Q = np.array([[[1, 0], [0, 1]]], dtype=np.float64)          # shape (1,2,2)
K = Q.copy()
V = np.array([[[1, 2], [3, 4]]], dtype=np.float64)

# Non‑causal attention
O_nc = scaled_dot_product_attention(Q, K, V, causal=False)
print(O_nc)   # [[... ...] [... ...]]

# Causal attention
O_c = scaled_dot_product_attention(Q, K, V, causal=True)
print(O_c)   # lower‑triangular influence only
```

The exact numerical values are omitted; the grader will compute them automatically.

## What the gate checks

* The grader computes a NumPy reference implementation for each test case.
* It compares your output to that reference using the `max_abs_err` scorer from `arena.scorers`.
* Your solution passes if the maximum absolute error across all test cases is at most $10^{-6}$.
