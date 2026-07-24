## Context

ZeRO optimizer stages reduce memory by partitioning optimizer states, gradients, and parameters across data parallel workers. Communication volume is often expressed relative to $\Phi$, the number of model parameters.

For ZeRO stage-2, the per-step communication consists of gradient reduce-scatter and parameter all-gather operations. The combined communication volume is

$$V_{stage2} = 2\Phi.$$

ZeRO stage-3 additionally partitions parameters, requiring an extra forward parameter all-gather. The per-step volume becomes

$$V_{stage3} = 3\Phi.$$

The value returned by the implementation should be the communication volume in units of $\Phi$, so it should not depend on the actual number of parameters.

## Task

Implement `compute_comm_volume(phi, stage)`:

```python
def compute_comm_volume(phi: float, stage: int) -> float:
    ...
```

The function takes the parameter count $\Phi$ and a ZeRO stage. It returns the total communication volume for one training step.

The returned value must be measured in the same units as `phi`. Valid stages are `2` and `3`.

For stage 2, include reduce-scatter and all-gather communication. For stage 3, include those operations plus the additional forward parameter all-gather.

Raise `ValueError` for unsupported stages.

## Example

```python
print(compute_comm_volume(1000000, 2))
# 2000000.0

print(compute_comm_volume(1000000, 3))
# 3000000.0
```

## What the gate checks

The gate computes an independent ZeRO communication oracle from the required collective operations and compares the returned values for several parameter counts and stages.

The `exact_match` score must be $1.0`. Incorrect stage handling or implementations that ignore $\Phi$ will fail.
