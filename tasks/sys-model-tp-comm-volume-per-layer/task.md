## Context

Tensor parallelism splits the computation of large transformer layers across $N$ devices. Some tensor-parallel linear operations require an all-reduce to combine partial results from every rank.

Assume a transformer layer has two tensor-parallel all-reduces. Each all-reduce communicates an activation tensor with shape $(b, s, h)$, where $b$ is batch size, $s$ is sequence length, and $h$ is hidden size. The communication volume is measured per rank using the ring all-reduce model.

For a tensor with $x$ elements, the ring all-reduce communication volume per rank is

$$x \cdot \frac{N-1}{N}.$$

The model uses bfloat16 activations, so each element occupies $2$ bytes. The total communication volume for one layer is therefore

$$
2 \cdot (bsh) \cdot \frac{N-1}{N} \cdot 2,
$$

where the first factor of $2$ is the number of all-reduce operations and the second factor of $2$ is the number of bytes per element.

## Task

Implement `tp_comm_volume_per_layer(b, s, h, N)`.

The function must return the modeled all-reduce communication bytes per transformer layer per rank as a Python float. The inputs are positive integers:

```python
def tp_comm_volume_per_layer(b: int, s: int, h: int, N: int) -> float:
    ...
```

Use the formula from the context. Do not round the result.

## Example

```python
bytes_per_layer = tp_comm_volume_per_layer(8, 2048, 4096, 8)
# 293601280.0
```

## What the gate checks

The gate computes the communication model independently with NumPy float64 arithmetic and compares the returned value against that oracle. The metric `modeled_mem_access` is `1.0` only when the implementation matches the model for all tested configurations.
