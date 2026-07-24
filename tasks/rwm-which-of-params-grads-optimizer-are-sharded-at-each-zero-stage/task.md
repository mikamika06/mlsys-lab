## Context

In distributed training, ZeRO (Zero Redundancy Optimizer) partitions the model state across data‑parallel workers to reduce memory usage.  
At each ZeRO stage a different subset of the three main components is sharded:

- **Parameters** – the weights of the neural network.
- **Gradients** – the per‑parameter gradients computed during back‑propagation.
- **Optimizer state** – e.g., momentum buffers, running averages.

The partitioning scheme follows a fixed progression:

| Stage | Parameters | Gradients | Optimizer |
|-------|------------|-----------|-----------|
| 0     | ❌         | ❌        | ❌        |
| 1     | ❌         | ❌        | ✅        |
| 2     | ❌         | ✅        | ✅        |
| 3     | ✅         | ✅        | ✅        |

Here “✅” means the component is sharded across workers, while “❌” indicates it remains replicated.

## Task

Implement a function that reports which components are sharded for a given ZeRO stage:

```python
def zero_stage_sharding(stage: int) -> tuple[bool, bool, bool]:
    ...
```

The returned tuple must be in the order `(params_sharded, grads_sharded, optimizer_sharded)` and contain Python `bool` values.  
The function should accept only integer stages 0–3 inclusive; other inputs may raise an exception.

## Example

```python
>>> zero_stage_sharding(0)
(False, False, False)

>>> zero_stage_sharding(1)
(False, False, True)

>>> zero_stage_sharding(2)
(False, True, True)

>>> zero_stage_sharding(3)
(True, True, True)
```

## What the gate checks

The grader verifies that the returned tuple matches the exact mapping above for all four stages.  The comparison is performed programmatically; no hard‑coded expected values are used in the reference implementation.
