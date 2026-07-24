## Context

A GPU kernel often operates on data stored in high bandwidth memory (HBM). When multiple operators are executed separately, intermediate tensors may be written back to HBM and read again by later operators.

Consider a matrix multiplication followed by bias addition and an activation function:

$$
Y = \sigma(XW + b),
$$

where $X \in \mathbb{R}^{m \times k}$, $W \in \mathbb{R}^{k \times n}$, and $b \in \mathbb{R}^{n}$.

In an unfused implementation, the matrix multiplication produces an intermediate tensor of shape $(m,n)$. That tensor is written to HBM, then read for the bias operation. The bias output is written again and read again for the activation. The final activation output is also written.

A fused implementation keeps intermediate values on-chip and only writes the final output. The modeled number of HBM tensor transfers is therefore lower.

For this task, one HBM round-trip means one full tensor movement to or from HBM. A tensor with $s$ elements contributes $s$ units for each read or write.

## Task

Implement `count_hbm_round_trips(m, k, n)`:

```python
def count_hbm_round_trips(m: int, k: int, n: int) -> dict:
    ...
```

Return a dictionary with the modeled HBM traffic for separate and fused execution:

```python
{
    "unfused": <number of HBM element transfers>,
    "fused": <number of HBM element transfers>
}
```

Assume all tensors use the same element size, so only element counts matter. The model counts:

- input matrix $X$ read once,
- weight matrix $W$ read once,
- bias vector $b$ read once,
- output tensors written when produced,
- intermediate tensors read again when consumed.

The final activation output is always written to HBM.

## Example

```python
result = count_hbm_round_trips(4, 8, 16)

# {
#   "unfused": 992,
#   "fused": 608
# }
```

The unfused calculation models separate matmul, bias, and activation kernels. The fused calculation models a single kernel that writes only the final result.

## What the gate checks

The gate recomputes the transfer model independently from the tensor dimensions and compares the returned dictionary exactly.

The metric `modeled_mem_access` is `1.0` only when the implementation matches the model for every tested matrix shape.
