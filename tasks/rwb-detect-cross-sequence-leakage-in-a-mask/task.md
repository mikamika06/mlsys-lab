## Context

In attention mechanisms for variable‑length sequences, a common pattern is to use a *block‑diagonal* mask that allows queries to attend only within their own sequence segment. Let $N$ be the total number of tokens and let $\text{cu\_seqlens}\in \mathbb{Z}^{B+1}$ be the cumulative sequence lengths, with $\text{cu\_seqlens}[0]=0$, $\text{cu\_seqlens}[B]=N$. The $i$‑th segment occupies indices
$$
[\text{cu\_seqlens}_i,\;\text{cu\_seqlens}_{i+1}) .
$$

A mask matrix $M \in \{0,1\}^{N\times N}$ is *valid* if for every pair $(q,k)$ with $M_{qk}=1$ we have
$$
q \text{ and } k \text{ belong to the same segment}.
$$
If any allowed pair crosses a boundary, we say that the mask **leaks**.

## Task

Implement `detect_leakage(mask, cu_seqlens)`:

```python
def detect_leakage(mask: np.ndarray, cu_seqlens: np.ndarray) -> bool:
    ...
```

It receives a 2‑D NumPy array of shape $(N,N)$ containing only 0/1 and the cumulative sequence lengths. Return `True` if the mask leaks (i.e., contains at least one allowed pair that crosses a segment boundary), otherwise return `False`. The implementation must be fully vectorized; no explicit Python loops over tokens.

## Example

```python
import numpy as np
mask = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,1],
                 [0,0,1,1]])
cu_seqlens = np.array([0,2,4])   # two segments: 0–1 and 2–3
print(detect_leakage(mask, cu_seqlens))  # False

mask[1,2] = 1   # allow query 1 to attend token 2 (cross‑segment)
print(detect_leakage(mask, cu_seqlens))  # True
```

## What the gate checks

The grader computes a NumPy oracle that implements the definition above and compares it with your function. The metric `exact_match` must equal $1.0$ for all test cases; any mismatch yields $0.0$. No other performance metrics are required.
