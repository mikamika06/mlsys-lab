## Context

Large distributed training systems split a model's parameters across data or model
parallel ranks. A flat partition assigns contiguous parameter ownership so that
every rank receives a nearly equal number of parameters.

For a model with $\Phi$ total parameters and $N$ ranks, the base number of
parameters per rank is

$$q = \left\lfloor \frac{\Phi}{N} \right\rfloor,$$

and the remaining parameters are

$$r = \Phi \bmod N.$$

The first $r$ ranks receive one extra parameter. Therefore rank $i$ owns

$$q + \begin{cases}
1 & \text{if } i < r,\\
0 & \text{otherwise}.
\end{cases}$$

For each owned parameter, the implementation tracks three buffers: parameters,
gradients, and optimizer state. If one parameter uses `param_bytes` bytes, one
gradient uses `grad_bytes` bytes, and optimizer state uses `opt_bytes` bytes, the
total bytes owned by a rank are

$$b_i = p_i \cdot (param\_bytes + grad\_bytes + opt\_bytes),$$

where $p_i$ is the parameter count assigned to rank $i$.

## Task

Implement `partition_phi(phi, n_ranks, param_bytes, grad_bytes, opt_bytes)`.

The function must return a Python list of length `n_ranks`. Each element is a
3-tuple:

```python
(param_bytes_owned, grad_bytes_owned, opt_bytes_owned)
```

for that rank.

The partition must assign `floor(phi / n_ranks)` parameters to every rank and
give the extra `phi % n_ranks` parameters to the lowest rank indices first.

All returned byte counts must be integers.

## Example

```python
result = partition_phi(10, 3, 4, 4, 8)

# Rank parameter counts are [4, 3, 3].
# Each parameter uses (4, 4, 8) bytes.
result == [
    (16, 16, 32),
    (12, 12, 24),
    (12, 12, 24),
]
```

## What the gate checks

The gate builds several cases and computes the expected ownership vector from
the partition algorithm itself. It compares the submitted function's complete
per-rank byte vector against this oracle with exact equality.
