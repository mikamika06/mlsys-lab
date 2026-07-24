## Context

ZeRO (Zero Redundancy Optimizer) removes the memory redundancy of plain
data parallelism, where every device holds a full replica of the
parameters, gradients, and optimizer states. For a model with $\Psi$
parameters trained with mixed-precision Adam, the *unsharded* per-replica
cost is:

$$
\underbrace{2\Psi}_{\text{fp16 params}} + \underbrace{2\Psi}_{\text{fp16 grads}} +
\underbrace{4\Psi + 4\Psi + 4\Psi}_{\text{fp32 master copy + momentum + variance}} = 16\Psi \text{ bytes}
$$

ZeRO progressively **partitions** these buffers across $N$ data-parallel
devices instead of replicating them, in three stages:

- **Stage 1** ($P_{os}$ — partition optimizer states only): parameters and
  gradients stay fully replicated; only the $12\Psi$ bytes of optimizer
  state are split $N$ ways.
  $$
  \text{bytes/device} = 4\Psi + \frac{12\Psi}{N}
  $$
- **Stage 2** ($P_{os+g}$ — also partition gradients): parameters stay
  replicated; both gradients and optimizer states are split $N$ ways.
  $$
  \text{bytes/device} = 2\Psi + \frac{14\Psi}{N}
  $$
- **Stage 3** ($P_{os+g+p}$ — also partition parameters): everything is
  split $N$ ways.
  $$
  \text{bytes/device} = \frac{16\Psi}{N}
  $$

## Task

Implement `zero_stage_bytes`:

```python
def zero_stage_bytes(num_params: int, num_devices: int, stage: int) -> float:
    ...
```

- `num_params`: $\Psi$, the number of model parameters.
- `num_devices`: $N$, the number of data-parallel devices.
- `stage`: `1`, `2`, or `3`.

Return the per-device memory footprint in bytes, as a `float`, using the
formula for the given stage above.

## Example

```python
zero_stage_bytes(num_params=7_000_000_000, num_devices=8, stage=1)
# 4 * 7e9 + 12 * 7e9 / 8 = 2.8e10 + 1.05e10 = 3.85e10 bytes

zero_stage_bytes(num_params=7_000_000_000, num_devices=8, stage=3)
# 16 * 7e9 / 8 = 1.4e10 bytes
```

## What the gate checks

The grader sweeps a range of `(num_params, num_devices, stage)`
combinations (including `num_devices == 1`, small models, and a
realistic 7B-parameter model) and computes the reference per-device
byte count directly from the formulas above.

`size_ratio` is the worst-case absolute deviation of `your_value /
expected_value` from `1.0`, across every combination tested (must be
`< 1e-9`). Using the wrong per-stage split (e.g. partitioning gradients
at stage 1, or forgetting to partition the parameter term at stage 3)
changes the ratio measurably and fails this gate.
