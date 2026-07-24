## Context

In deep learning, a *pruned* model contains many weight parameters that have been set to exactly zero in order to reduce memory usage and inference cost.  
The **realized sparsity** of a tensor $W \in \mathbb{R}^{n_1\times\cdots\times n_k}$ is defined as

$$
s(W) \;=\;\frac{\#\{\,i \mid W_i = 0\,\}}{\operatorname{size}(W)} ,
$$

where $\operatorname{size}(W)=\prod_{j=1}^k n_j$ is the total number of elements.  
The numerator counts the exact zeros, while the denominator counts all entries.

## Task

Implement `count_zeros_and_sparsity(W)`:

```python
def count_zeros_and_sparsity(W: np.ndarray) -> tuple[int, float]:
    ...
```

It receives a NumPy array of arbitrary shape and returns a two‑tuple:
1. The integer number of elements equal to zero.
2. The realized sparsity ratio as a `float`.

The function must work for any numeric dtype and should not modify the input tensor.

## Example

```python
import numpy as np
W = np.array([[0, 3], [4, 0]])
num_zeros, sparsity = count_zeros_and_sparsity(W)
print(num_zeros)   # 2
print(sparsity)    # 0.5
```

## What the gate checks

The grader computes the reference answer with NumPy on a set of test tensors and compares it to your output using an exact match metric. Your implementation must return exactly the same integer count and floating‑point ratio (within machine precision). Any deviation causes the gate to fail.
