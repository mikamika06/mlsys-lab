## Context

Finite-precision arithmetic introduces rounding errors that accumulate with the
number of operations. When summing many small numbers, the error growth depends
on the accumulator's precision. Half precision (FP16) has only 11 bits of
mantissa, while single precision (FP32) has 24 bits.

Let $x_i$ be random numbers in $[-1, 1]$. Define the partial sums

$$
s_k^{(p)} = \sum_{i=1}^{k} x_i
$$

computed in precision $p \in \{\text{fp16}, \text{fp32}\}$, each cast back to
`float64` for comparison. The *accumulation error* at step $k$ is

$$
e_k^{(p)} = \bigl| s_k^{(p)} - s_k^{(\text{ref})} \bigr|
$$

where $s_k^{(\text{ref})}$ is the exact sum in `float64`. The error typically
grows roughly as $\sqrt{k}$ for random signs, but the magnitude depends on the
precision.

## Task

Implement

```python
def accum_error_growth(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate accumulation of n random numbers in fp16 and fp32.
    Returns (err16, err32) arrays of length n with absolute error at each step.
    """
```

Generate `n` random numbers uniformly in `[-1, 1]` using NumPy with the given
seed. Compute cumulative sums in both FP16 and FP32 accumulators, comparing each
to the exact FP64 cumulative sum. Return two `float64` arrays of absolute errors
for FP16 and FP32 respectively.

## Example

```python
import numpy as np
err16, err32 = accum_error_growth(5, seed=1)
print(np.round(err16, 6))
print(np.round(err32, 6))
# Example output (values vary slightly):
# [0.       0.000122 0.000244 0.000366 0.000488]
# [0.       0.       0.       0.       0.      ]
```

## What the gate checks

The grader recomputes the reference error curves deterministically and compares
your arrays with a relative error metric

$$
\mathrm{rel\_err} = \frac{\lVert \hat{e} - e_{\text{ref}} \rVert_2}{\lVert e_{\text{ref}} \rVert_2}.
$$

Your implementation must match the reference within $10^{-6}$ relative error.
