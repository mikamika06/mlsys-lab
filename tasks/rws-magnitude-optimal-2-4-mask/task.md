## Context

In many neural‑network compression schemes a *2:4 sparsity* pattern is used: for every consecutive block of four weights, exactly two are kept and the other two are set to zero.  
A common strategy is to keep the two largest‑magnitude weights in each block, which maximises the retained signal energy.

Let $W \in \mathbb{R}^{n\times m}$ be a weight matrix with $m$ divisible by 4.  
For each block of four consecutive columns we define

$$
\text{mask}_{i,j} = 
\begin{cases}
1 & \text{if } |W_{i,4j+r}| \text{ is among the two largest in its block},\\[4pt]
0 & \text{otherwise},
\end{cases}
$$

where $r\in\{0,1,2,3\}$ indexes the position inside the block.  
The resulting binary mask has exactly two ones per block and zeros elsewhere.

## Task

Implement `magnitude_optimal_2to4_mask`:

```python
def magnitude_optimal_2to4_mask(weights: np.ndarray) -> np.ndarray:
    ...
```

It receives a 2‑D NumPy array of shape $(n,m)$ with $m$ a multiple of 4 and returns a boolean mask of the same shape.  
The mask must satisfy:

1. Exactly two `True` values per consecutive group of four columns.
2. The sum $\sum_{i,j} |W_{ij}|\,\text{mask}_{ij}$ is maximal among all valid 2:4 masks.

Use only vectorised NumPy operations; no explicit Python loops over elements or groups.

## Example

```python
import numpy as np
weights = np.array([[0.1, -3.5, 2.0, 0.0,
                     1.2,  4.8, -0.7, 0.9],
                    [5.0, -1.1, 0.3, 2.2,
                     -4.4, 0.6, 3.3, -2.2]])
mask = magnitude_optimal_2to4_mask(weights)
print(mask.astype(int))
# [[0 1 1 0
#   1 1 0 0]
#  [1 0 0 1
#   0 0 1 1]]
```

The mask keeps the two largest magnitudes in each block.

## What the gate checks

The grader computes a reference mask with NumPy’s `argpartition` and verifies that your output matches it exactly.  
It also confirms that the mask has the correct shape, dtype (`bool`), and that the retained magnitude sum equals the maximum possible within $10^{-9}$ relative error.
