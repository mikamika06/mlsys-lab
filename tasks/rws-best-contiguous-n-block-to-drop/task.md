## Context

Transformer and other deep learning systems often store hidden states at layer boundaries. If a contiguous block of layers can be removed, a useful heuristic is to measure how much the representation changes when jumping over the block.

For two non-zero vectors $a,b \in \mathbb{R}^d$, the angular distance is

$$
\theta(a,b)=\arccos\left(\frac{a^\top b}{\lVert a\rVert_2\lVert b\rVert_2}\right).
$$

Given hidden states $H \in \mathbb{R}^{B \times L \times d}$, where $B$ is the batch size and $L$ is the number of layer boundaries, deleting a block of $n$ layers starting at layer $s$ is approximated by comparing $H[:,s,:]$ with $H[:,s+n,:]$.

The score for a start position is the mean angular distance over the batch:

$$
score(s)=\frac{1}{B}\sum_{i=1}^{B}\theta(H_{i,s},H_{i,s+n}).
$$

The best block is the one with the smallest score.

## Task

Implement `best_contiguous_n_block_to_drop`:

```python
def best_contiguous_n_block_to_drop(
    hidden_states: np.ndarray, n: int
) -> tuple[int, float]:
    ...
```

`hidden_states` is a NumPy array with shape $(B,L,d)$. The integer `n` is the number of consecutive layers to skip. Return a tuple containing:

1. the start layer index $s$ with the minimum `score(s)`;
2. the corresponding mean angular distance as a Python `float`.

Use NumPy operations for the computation. The input vectors are non-zero. If multiple positions have the same minimum score, return the smallest index.

## Example

```python
import numpy as np

H = np.array([
    [[1, 0], [0.99, 0.01], [0, 1]],
])

index, distance = best_contiguous_n_block_to_drop(H, 1)

# index == 0 because H[:,0,:] and H[:,1,:] have the smallest angle
```

## What the gate checks

The gate computes an independent NumPy oracle by sweeping every valid start position and calculating the mean angular distance.

The `argmin_index` metric requires the returned start index to exactly match the oracle. The `distance_error` metric requires the returned distance to differ from the oracle by no more than $10^{-6}$.
