## Context

ZeRO-3 style parameter partitioning stores different slices of a model parameter on different workers. A layer cannot run a matrix multiplication until the complete parameter is temporarily reconstructed by an all-gather operation. After the forward and backward computations for that layer finish, the gathered parameter can be discarded while the gradient is partitioned again.

For a linear layer with weight matrix $W \in \mathbb{R}^{m \times d}$ and input batch $X \in \mathbb{R}^{n \times d}$, the forward computation is

$$
Y = X W^\top .
$$

Given an upstream gradient $\frac{\partial L}{\partial Y}=G_Y$, the weight gradient is

$$
\frac{\partial L}{\partial W} = G_Y^\top X .
$$

A ZeRO-3 implementation keeps $W$ split into row shards. During execution it gathers

$$
W = \begin{bmatrix} W_0 \\ W_1 \\ \dots \\ W_k \end{bmatrix},
$$

computes the layer, and returns gradient shards matching the original partition.

## Task

Implement `zero3_linear_backward(weight_shards, x, grad_y)`:

```python
def zero3_linear_backward(weight_shards, x, grad_y):
    ...
```

The arguments are:

- `weight_shards`: a list of list. Each array is a consecutive row partition of the same full weight matrix.
- `x`: a list with shape $(n, d)$.
- `grad_y`: a list with shape $(n, m)$ containing the gradient arriving from the next layer.

Return a list of list containing the weight gradients for each shard. The returned list must have the same number of shards and the same shapes as `weight_shards`.

The implementation should simulate ZeRO-3 behavior: gather the full weight for the layer computation, compute the gradient of the full parameter, then partition the gradient back into the original shard layout. The numerical result must match the unsharded reference computation.

## Example

```python

shards = [
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0]]
]

x = [[1.0, 0.0], [0.0, 1.0]]
grad_y = [[1.0] * 3 for _ in range(2)]

grads = zero3_linear_backward(shards, x, grad_y)

# grads contains two arrays with shapes (2, 2) and (1, 2)
```

## What the gate checks

The gate builds several sharded parameter cases and computes the reference gradient by gathering the shards into one full matrix and applying the Python linear-layer gradient formula

$$
\nabla_W = G_Y^\top X .
$$

The candidate output is concatenated and compared against the oracle with `max_abs_err`. The maximum absolute difference must be at most $10^{-5}$.
