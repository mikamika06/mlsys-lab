## Context

A transformer attention block creates query, key, and value matrices from the same input activations.

For an input matrix $X \in \mathbb{R}^{n \times d}$, the three projections are traditionally computed as

$$
Q = XW_q,\qquad K = XW_k,\qquad V = XW_v ,
$$

where $W_q, W_k, W_v \in \mathbb{R}^{d \times m}$.

The three matrix multiplications read the same input and perform the same kind of operation. They can be fused by concatenating the projection weights:

$$
W_{qkv} = [W_q \; W_k \; W_v] .
$$

Then one multiplication produces all three outputs:

$$
QKV = XW_{qkv}.
$$

The result can be split into three views without additional matrix multiplications.

## Task

Implement `fused_qkv_projection(X, Wq, Wk, Wv)`:

```python
def fused_qkv_projection(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

The function receives an input activation matrix $X$ and three projection matrices. Each
projection matrix has shape $(d, m)$. Return $(Q, K, V)$ where each matrix has shape
$(n, m)$.

The implementation must compute the three projections using one fused matrix
multiplication followed by slicing. Do not call matrix multiplication separately for
$W_q$, $W_k$, and $W_v$.

The returned arrays must be numerically equivalent to the separate projection formula
using float64 NumPy operations.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0]])
Wq = np.array([[1.0], [0.0]])
Wk = np.array([[0.0], [1.0]])
Wv = np.array([[1.0], [1.0]])

Q, K, V = fused_qkv_projection(X, Wq, Wk, Wv)

# Q = [[1.0]]
# K = [[2.0]]
# V = [[3.0]]
```

## What the gate checks

The numeric gate computes the reference answer with NumPy using the unfused definition
$Q=XW_q$, $K=XW_k$, and $V=XW_v$. The maximum absolute error must satisfy
$\max |A-B| \le 10^{-6}$.

The optimization gate instruments NumPy matrix multiplication calls while running the
candidate implementation. The candidate must use exactly one matrix multiplication.
An implementation with three independent projections fails this gate even if the
numeric output is correct.
