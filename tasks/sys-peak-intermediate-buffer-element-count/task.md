## Context

Softmax converts a vector of logits $x \in \mathbb{R}^N$ into a probability vector:

$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} .
$$

A direct implementation stores all intermediate exponentials:

$$
e_i = e^{x_i}, \qquad s = \sum_i e_i ,
$$

which requires an intermediate buffer of size $O(N)$. For long sequences or large batches, this temporary storage can dominate memory use.

An online softmax computes the result while streaming through the input. It maintains a running maximum $m$ and running exponential sum $s$:

$$
m' = \max(m, x_i)
$$

and

$$
s' = s e^{m-m'} + e^{x_i-m'} .
$$

After processing all elements, the normalization uses the final values of $m$ and $s$. This allows the algorithm to use memory proportional to the current row width rather than the full sequence length.

The task models memory usage by asking the implementation to report its peak number of simultaneously live intermediate elements. The count is a model of buffers, not a measurement of the Python process heap.

## Task

Implement `online_softmax_stream(x, B)`:

```python
def online_softmax_stream(x: np.ndarray, B: int) -> tuple[np.ndarray, int]:
    ...
```

The input `x` is a one-dimensional NumPy array of length $N$. `B` is a block size. Return:

1. The softmax result as a `float64` NumPy array of length $N`.
2. An integer `peak_elements` representing the maximum number of live intermediate-buffer elements used by the algorithm.

Use a streaming algorithm. The implementation should not allocate buffers whose size grows with $N$. The reported memory count should represent the largest temporary storage used by the algorithm, including block-sized temporaries.

## Example

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y, peak = online_softmax_stream(x, 2)

# y is close to:
# [0.0320586, 0.0871443, 0.2368828, 0.6439143]

# peak should depend on the block size, not on len(x)
```

## What the gate checks

The gate compares the returned probabilities against a NumPy softmax reference computed by the grader.

It also checks the reported `peak_elements` value. Across inputs with different lengths $N$, the value must remain bounded by a constant multiple of $d+B$, where $d=1$ for this one-dimensional task and $B$ is the block size. A solution that materializes an $N$-element exponential buffer and reports that size will fail because the count grows with $N$.
