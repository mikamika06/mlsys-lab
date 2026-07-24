## Context

Scaled dot-product attention computes how strongly each query attends to each key.
For queries $Q \in \mathbb{R}^{n \times d}$ and keys $K \in \mathbb{R}^{m \times d}$,
the attention logits are

$$
L = \frac{QK^\top}{\sqrt{d}} .
$$

The scale factor $1/\sqrt{d}$ keeps the magnitude of logits stable as the head
dimension $d$ grows. The attention weights are obtained with softmax:

$$
P_{ij} = \frac{e^{L_{ij}}}{\sum_{t=1}^{m} e^{L_{it}}}.
$$

The output is then

$$
O = PV ,
$$

where $V \in \mathbb{R}^{m \times d_v}$ contains the values.

A missing scale changes the softmax distribution because the logits become too
large. This causes attention to become overly concentrated and produces incorrect
outputs.

## Task

Implement `scaled_dot_product_attention(q, k, v)`:

```python
def scaled_dot_product_attention(q: np.ndarray,
                                 k: np.ndarray,
                                 v: np.ndarray) -> np.ndarray:
    ...
```

The inputs are two-dimensional NumPy arrays:

- `q` has shape $(n, d)$.
- `k` has shape $(m, d)$.
- `v` has shape $(m, d_v)$.

Return the attention output with shape $(n, d_v)$ using float64 computation.

The implementation must apply the scale factor $1/\sqrt{d}$ before softmax. Use a numerically stable softmax by subtracting the row maximum before exponentiation.

## Example

```python
import numpy as np

q = np.array([[1.0, 0.0]])
k = np.array([[1.0, 0.0], [0.0, 1.0]])
v = np.array([[2.0, 0.0], [0.0, 3.0]])

out = scaled_dot_product_attention(q, k, v)
```

The first query matches the first key more strongly, but the output remains a weighted
combination because softmax is applied to scaled logits.

## What the gate checks

The gate computes a NumPy float64 oracle implementation of scaled dot-product
attention and compares the candidate output using maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |O_i - \hat{O}_i|.
$$

The result must satisfy $\mathrm{max\_abs\_err} < 10^{-6}$. Implementations that
omit $1/\sqrt{d}$ or multiply logits by $\sqrt{d}$ produce a different softmax
distribution and fail the numerical check.
