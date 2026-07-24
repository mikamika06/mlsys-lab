## Context

Data parallel (DP) training keeps a full copy of model parameters on every worker. For a model with $P$ parameters and element size $b$ bytes, the gradient synchronization communication per training step can be modeled as an all-reduce over the full gradient tensor.

A ring all-reduce moves approximately

$$
2 \frac{W-1}{W} \cdot P \cdot b
$$

bytes per worker, where $W$ is the number of workers.

ZeRO stage 3 shards parameters and gradients across workers. Instead of keeping a full model replica, each step gathers parameter shards for computation and reduces gradients back into shards. A simplified communication model is:

$$
\text{ZeRO-3 bytes}
=
2 \cdot \frac{W-1}{W} \cdot P \cdot b ,
$$

where the first term represents parameter all-gather and the second term represents gradient reduce-scatter.

The comparison is about the communication volume model, not actual network scheduling.

## Task

Implement `compare_zero3_dp_comm(params, world_size, bytes_per_param)`:

```python
def compare_zero3_dp_comm(
    params: int,
    world_size: int,
    bytes_per_param: int,
) -> dict:
    ...
```

Return a dictionary with exactly these keys:

- `"dp_bytes"`: modeled communication bytes per worker for plain data parallel training.
- `"zero3_bytes"`: modeled communication bytes per worker for ZeRO-3.
- `"ratio"`: `zero3_bytes / dp_bytes`.

All returned numeric values must be Python `float` values.

Use the formulas from the context. Inputs satisfy $params > 0$, $world\_size > 1$, and $bytes\_per\_param > 0$.

## Example

```python
result = compare_zero3_dp_comm(1000000, 8, 2)

# result:
# {
#   "dp_bytes": 3500000.0,
#   "zero3_bytes": 3500000.0,
#   "ratio": 1.0,
# }
```

## What the gate checks

The gate computes the reference model independently using NumPy arithmetic and compares the returned values on several parameter sizes, worker counts, and parameter byte widths.

The metric `modeled_mem_access` is `1.0` only when all modeled communication values match the oracle output exactly.
