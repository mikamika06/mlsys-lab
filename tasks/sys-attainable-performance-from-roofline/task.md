## Context

The roofline model gives an upper bound on the attainable performance of a kernel in terms of its arithmetic intensity $I$ (flops per byte) and the hardware limits: peak floating‑point throughput $\mathrm{Peak}_{\text{FLOP}}$ and memory bandwidth $\mathrm{BW}$.  
For a given configuration the attainable performance is

$$
P_{\max}(I)=\min \bigl(\mathrm{Peak}_{\text{FLOP}},\, I \times \mathrm{BW}\bigr).
$$

This simple formula captures the fact that a kernel can be either compute‑bound or memory‑bound.

## Task

Implement `roofline_perf(ai, peak_flops, mem_bandwidth)`:

```python
def roofline_perf(ai: np.ndarray | float,
                  peak_flops: np.ndarray | float,
                  mem_bandwidth: np.ndarray | float) -> np.ndarray:
    ...
```

The arguments may be scalars or NumPy arrays of the same shape.  
Return an array (or scalar) containing the attainable performance for each element, computed with the formula above.  Use only vectorised NumPy operations; no Python loops.

## Example

```python
import numpy as np
ai = np.array([1.0, 2.5])
peak = np.array([10.0, 20.0])
bw   = np.array([4.0, 3.0])

P = roofline_perf(ai, peak, bw)
# P == [min(10, 1*4), min(20, 2.5*3)] -> array([4., 7.5])
```

## What the gate checks

The grader evaluates a set of random test cases and compares your result to a reference computed by NumPy’s `minimum`.  
It reports the global relative L2 error

$$
\mathrm{rel\_err} = \frac{\lVert P_{\text{got}}-P_{\text{ref}}\rVert}
                        {\lVert P_{\text{ref}}\rVert + 10^{-12}},
$$

which must be $\le 10^{-9}$.
