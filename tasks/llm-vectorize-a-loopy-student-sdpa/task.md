## Context

Scaled dot-product attention computes a weighted combination of value vectors using
query-key similarity scores.

For query matrix $Q \in \mathbb{R}^{n \times d}$, key matrix $K \in \mathbb{R}^{n \times d}$,
and value matrix $V \in \mathbb{R}^{n \times m}$, the attention scores are

$$
S = \frac{QK^\top}{\sqrt{d}} .
$$

The softmax operation is applied independently to every row:

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}.
$$

The final scaled dot-product attention output is

$$
O = PV .
$$

A direct implementation can use three nested Python loops over queries, keys, and
value dimensions. This repeats many small operations in the Python interpreter.
NumPy can compute the same expression using optimized vectorized operations.

## Task

Implement `sdpa(Q, K, V)`:

```python
def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    ...
```

The inputs are floating point NumPy arrays with shapes $(n,d)$, $(n,d)$, and
$(n,m)$. Return the attention output with shape $(n,m)$ as `float64`.

The implementation must be vectorized NumPy code. Do not use Python loops over
tokens or dimensions.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 3.0], [4.0, 5.0]])

O = sdpa(Q, K, V)
# Each row is a softmax-weighted mixture of V rows.
```

## What the gate checks

The numeric gate computes the reference attention output using a NumPy oracle and
requires

$$
\max_{i,j} |O_{ij}^{candidate} - O_{ij}^{reference}| \le 10^{-6}.
$$

The vectorization gate records Python line execution events inside the submitted
function using `sys.settrace`. The count must remain below `20`. A triple nested
Python implementation produces many line events and fails, while a small
vectorized implementation passes.
