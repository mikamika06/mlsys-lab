## Context

In transformer‑style attention a *causal* (or *look‑ahead*) mask is used to forbid each token from attending to future tokens.  
For a sequence of length $n$ the mask $M \in \mathbb{R}^{n\times n}$ has entries

$$
M_{ij} = 
\begin{cases}
1 & \text{if } j \le i,\\[4pt]
0 & \text{otherwise},
\end{cases}
$$

so that the $i$‑th query can only attend to positions $\le i$.  
In NumPy this is exactly the lower triangular part of an all‑ones matrix:  

$$
M = \operatorname{tril}\!\bigl(\mathbf{1}_{n\times n}\bigr).
$$

A common off‑by‑one bug occurs when the implementation mistakenly allows the token at position $i+1$ to be attended, i.e. it sets $M_{i,i+1}=1$.  This *future leak* can silently degrade model performance.

## Task

Implement a function that returns the correct causal mask:

```python
def create_causal_mask(n: int) -> np.ndarray:
    ...
```

The function receives an integer sequence length `n` and must return an `(n, n)` NumPy array of type `float64`.  The returned matrix should have ones on and below the main diagonal and zeros elsewhere.

## Example

```python
import numpy as np
mask = create_causal_mask(4)
print(mask)
# [[1. 0. 0. 0.]
#  [1. 1. 0. 0.]
#  [1. 1. 1. 0.]
#  [1. 1. 1. 1.]]
```

## What the gate checks

The grader computes a reference mask with `np.tril(np.ones((n, n), dtype=np.float64))` for several sequence lengths and evaluates the **maximum absolute error** between the candidate output and this reference:

$$
\mathrm{max\_abs\_err} = \max_{i,j}\bigl|\,M^{\text{cand}}_{ij}-M^{\text{ref}}_{ij}\bigr|.
$$

The solution must achieve `max_abs_err ≤ 1e-6`.  Any leakage of a future token will produce an error of at least `1.0` and fail the gate.
