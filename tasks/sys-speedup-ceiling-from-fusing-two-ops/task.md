## Context

A roofline-style performance estimate can start from memory traffic. For a pair of
elementwise operations, an unfused implementation may materialize an intermediate
array in memory.

Consider the computation

$$
z = x \odot y,\qquad out = z + 1,
$$

where $\odot$ is elementwise multiplication. Without fusion, the intermediate $z$
must be written and then read again:

$$
B_{\mathrm{unfused}} =
\mathrm{bytes}(x) + \mathrm{bytes}(y) + \mathrm{bytes}(z)
+ \mathrm{bytes}(z) + \mathrm{bytes}(out).
$$

With fusion, the intermediate is kept in registers or cache and the estimated
traffic becomes

$$
B_{\mathrm{fused}} =
\mathrm{bytes}(x) + \mathrm{bytes}(y) + \mathrm{bytes}(out).
$$

The memory-traffic speedup ceiling is the ratio

$$
S = \frac{B_{\mathrm{unfused}}}{B_{\mathrm{fused}}}.
$$

This is only a ceiling because real hardware also depends on arithmetic intensity,
cache behavior, instruction throughput, and other effects.

## Task

Implement `fused_elementwise_speedup(x, y)`.

The function receives two list with identical shape and dtype. It must
return the predicted speedup ceiling $S$ from the memory traffic model above as
a Python `float`.

Use the actual array byte sizes from Python. Do not inspect values or benchmark the
operations.

## Example

```python

x = [[0.0] * 1024 for _ in range(1024)]
y = [[1.0] * 1024 for _ in range(1024)]

speedup = fused_elementwise_speedup(x, y)
# The result is approximately 1.3333333333333333
```

## What the gate checks

The gate builds several list with different shapes and dtypes. It computes
the expected traffic ratio from the real list sizes and checks that the
returned `size_ratio` matches that oracle-derived value.

A solution that only returns a constant or assumes a fixed dtype will fail.
