## Context

IEEE floating-point numbers include very small subnormal values between zero and the smallest normal number. Some processors support flush-to-zero (FTZ), where subnormal intermediate values are replaced by zero to improve performance.

For a dot product,

$$
s = \sum_{i=1}^{n} a_i b_i ,
$$

the exact path preserves subnormal intermediate values while an FTZ path replaces products that are too small to represent as normal floating-point values with zero before accumulation.

This can change results even though both computations follow the same mathematical expression. The task uses a deterministic cache simulator as a second observable effect of the implementation: the memory access trace is simulated with fixed cache parameters rather than measured on real hardware.

## Task

Implement `dot_ftz_trace(a, b)`:

```python
def dot_ftz_trace(a: np.ndarray, b: np.ndarray) -> tuple[float, list[int]]:
    ...
```

The inputs are one-dimensional NumPy arrays of `float64` values with equal length. Return a pair:

1. A floating-point dot product computed with flush-to-zero behavior.
2. A byte-address access trace describing the memory accesses performed by the kernel.

The FTZ rule is:

$$
x_{\mathrm{ftz}} =
\begin{cases}
0 & \text{if } 0 < |x| < 2^{-1022}, \\
x & \text{otherwise.}
\end{cases}
$$

Apply this rule to each product $a_i b_i$ before adding it to the accumulator.

The trace must contain byte addresses for sequential reads of the two input arrays. The first array starts at address $0$ and the second array starts at address $4096$. Each `float64` element occupies $8$ bytes. Use one address per element read.

## Example

```python
import numpy as np

a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])

value, trace = dot_ftz_trace(a, b)

# value is 11.0
# trace is [0, 8, 4096, 4104]
```

## What the gate checks

The gate computes the exact reference implementation itself and compares the returned value with the FTZ reference using

$$
\mathrm{max\_abs\_err} = \max_i |x_i-y_i|.
$$

The returned value must match the deterministic FTZ computation.

The returned access trace is passed through a fixed cache simulator with line size $64$ bytes, $4$ sets, and $2$ ways. The simulator's miss count must match the reference access pattern. No real hardware timing or cache measurements are used.
