## Context

Tensor parallelism splits a model layer across multiple devices. A row-parallel linear
layer partitions the input dimension of a weight matrix across workers.

For a normal linear layer,

$$
Y = XW + b,
$$

where $X \in \mathbb{R}^{m \times d}$, $W \in \mathbb{R}^{d \times h}$, and
$Y \in \mathbb{R}^{m \times h}$.

In row parallelism, the weight matrix is split by rows:

$$
W =
\begin{bmatrix}
W_0 \\
W_1 \\
\vdots \\
W_{k-1}
\end{bmatrix},
$$

and the matching input slices are split:

$$
X =
\begin{bmatrix}
X_0 & X_1 & \dots & X_{k-1}
\end{bmatrix}.
$$

Each worker computes a partial output:

$$
Y_i = X_i W_i .
$$

The complete result requires an all-reduce sum across workers:

$$
Y = \sum_{i=0}^{k-1} Y_i + b .
$$

Returning only one worker's partial result is incorrect because it omits contributions
from the other partitions.

## Task

Implement `row_parallel_linear(x_shards, w_shards, bias)`.

Arguments:

- `x_shards` is a list of NumPy arrays. Each array has shape $(m, d_i)$.
- `w_shards` is a list of NumPy arrays. Each array has shape $(d_i, h)$.
- `bias` is a one-dimensional NumPy array with shape $(h,)$.

Return the full output of the row-parallel linear layer as a `float64` NumPy array.

The function must combine all worker partial outputs by summing them and then adding
the bias.

## Example

```python
import numpy as np

x_shards = [
    np.array([[1., 2.]]),
    np.array([[3., 4.]])
]
w_shards = [
    np.array([[1., 0.], [0., 1.]]),
    np.array([[2., 0.], [0., 2.]])
]
bias = np.array([1., 1.])

y = row_parallel_linear(x_shards, w_shards, bias)
# [[9., 11.]]
```

## What the gate checks

The gate builds several row-parallel layers and computes the unsharded reference result
by reconstructing the full weight matrix and evaluating the NumPy linear operation.

The returned result is compared with the oracle using relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}} - y_{\mathrm{oracle}} \rVert_2}
{\lVert y_{\mathrm{oracle}} \rVert_2 + 10^{-12}} .
$$

The result must satisfy $\mathrm{rel\_err} \le 10^{-5}$. A solution that returns only
one worker's partial sum fails because it does not perform the required all-reduce.
