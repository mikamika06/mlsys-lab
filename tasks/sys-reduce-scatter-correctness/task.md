## Context

A reduce-scatter collective combines two communication patterns. First, every rank contributes data that is reduced element-wise across ranks. Then, the resulting tensor is split so each rank receives only one chunk.

Assume there are $p$ ranks. Each rank $r$ stores $p$ chunks:

$$
X_r = [x_{r,0}, x_{r,1}, \dots, x_{r,p-1}],
$$

where each chunk $x_{r,i}$ is a vector of the same length. The global reduction computes:

$$
s_i = \sum_{r=0}^{p-1} x_{r,i}.
$$

After the reduction, rank $i$ owns only chunk $s_i$. This reduces the amount of data stored by each rank while preserving the result of the full reduction.

## Task

Implement `reduce_scatter_sum(chunks)`:

```python
def reduce_scatter_sum(chunks):
    ...
```

The input is a list of length $p$. Element `chunks[r]` is the data held by rank $r` and must be a list of $p$ NumPy arrays:

```python
chunks[r][i]
```

is rank $r$'s contribution to output chunk $i$.

Return a list of length $p`. The element at index `i` must be the reduced chunk owned by rank `i`:

```python
result[i] == sum(chunks[r][i] for r in range(p))
```

The implementation should support floating point NumPy arrays and preserve the numerical result of the reduction.

## Example

```python
import numpy as np

chunks = [
    [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
]

result = reduce_scatter_sum(chunks)

# result[0] is [6.0, 8.0]
# result[1] is [10.0, 12.0]
```

## What the gate checks

The gate builds several reduce-scatter inputs and computes the expected result using NumPy element-wise summation. It compares the submitted implementation with this oracle using the maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |y_i - \hat{y}_i|.
$$

The value must be at most $10^{-6}$. Implementations that return local chunks without reduction or assign reduced chunks to the wrong ranks will fail.
