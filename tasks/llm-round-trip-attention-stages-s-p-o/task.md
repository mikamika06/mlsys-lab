## Context

Scaled dot‑product attention is a core component of transformer models.  
Given query, key and value matrices $Q,K,V \in \mathbb{R}^{n\times d}$ the
attention mechanism computes

$$S = \frac{QK^\top}{\sqrt{d}},$$

then applies a row‑wise softmax to obtain probabilities

$$P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})},$$

and finally produces the output

$$O = PV.$$

The matrices $S$, $P$ and $O$ are all of shape $(n,n)$, $(n,n)$ and
$(n,d)$ respectively.

## Task

Implement a function that returns each of these intermediate stages:

```python
def attention_roundtrip(Q: np.ndarray,
                        K: np.ndarray,
                        V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the scaled dot‑product attention and return the three stages:
        S – raw scores before softmax (float64)
        P – row‑wise softmax probabilities (float64)
        O – weighted sum of values (float64)

    Parameters
    ----------
    Q, K, V : np.ndarray
        2‑D arrays of shape (n, d) with dtype float64.

    Returns
    -------
    S, P, O : tuple[np.ndarray, np.ndarray, np.ndarray]
        Each array has the same dtype as the inputs.
    """
```

The implementation must use only NumPy operations; no explicit Python loops are allowed.  
All returned arrays should be of type `float64`.

## Example

```python
import numpy as np
Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.]])
V = np.array([[2., 3.], [4., 5.]])
S, P, O = attention_roundtrip(Q, K, V)
print(S)   # [[1. 0.]
           #  [0. 1.]]
print(P)   # [[0.73105858 0.26894142]
           #  [0.26894142 0.73105858]]
print(O)   # [[2. 3.]
           #  [4. 5.]]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares
each stage with the student's output using the `max_abs_err` scorer.
Three separate gates are applied:

* `S`: maximum absolute difference between the student’s scores and the reference must be ≤ $10^{-6}$.
* `P`: same requirement for the softmax probabilities.
* `O`: same requirement for the final weighted sum.

These thresholds guarantee that the implementation is numerically
identical to the canonical NumPy version.
