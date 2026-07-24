## Context

The roofline model classifies a computation by comparing its arithmetic intensity to the machine balance point. Arithmetic intensity measures how much computation is performed per byte moved:

$$AI = \frac{\text{FLOPs}}{\text{bytes accessed}}.$$

A machine with peak compute throughput $P$ FLOP/s and memory bandwidth $B$ bytes/s has a balance point

$$AI^* = \frac{P}{B}.$$

An operation is compute-bound when its arithmetic intensity is high enough that compute throughput limits performance:

$$AI \ge AI^*.$$

Otherwise, memory movement is the limiting factor and the operation is memory-bound.

## Task

Implement `classify_roofline_ops(ops, peak_flops, bandwidth)`.

The argument `ops` is a list of dictionaries. Each dictionary contains:

- `"name"`: the operation name.
- `"flops"`: the estimated floating point operations.
- `"bytes"`: the estimated bytes transferred.

`peak_flops` is the machine peak throughput in FLOP/s and `bandwidth` is the memory bandwidth in bytes/s.

Return a list of dictionaries with the same order as `ops`. Each output dictionary must contain:

- `"name"`: copied from the input operation.
- `"ai"`: arithmetic intensity as a Python float.
- `"bound"`: either `"compute"` or `"memory"`.

Use the roofline rule described above. Do not round the arithmetic intensity.

## Example

```python
ops = [
    {"name": "matmul", "flops": 1000000, "bytes": 10000},
    {"name": "copy", "flops": 1000, "bytes": 100000},
]

out = classify_roofline_ops(ops, peak_flops=1e12, bandwidth=1e11)

# [
#   {"name": "matmul", "ai": 100.0, "bound": "compute"},
#   {"name": "copy", "ai": 0.01, "bound": "memory"}
# ]
```

## What the gate checks

The gate creates several operations and computes the expected classifications from the roofline equation using a NumPy-based reference calculation. The returned list must match the oracle exactly.
