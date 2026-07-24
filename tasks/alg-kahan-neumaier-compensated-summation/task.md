## Context
Floating-point addition is not strictly associative due to round-off errors. When accumulating many small numbers or adding a small number to a very large number, precision is lost. For example, in single-precision (`float32`), `1e13 + 1.0` results in `1e13`. 
To mitigate this, compensated summation techniques preserve the lost low-order bits. The **Kahan summation** algorithm keeps a running compensation term, but it can fail if the summands are highly variable in magnitude or alternating in sign. **Neumaier's modification** (also known as Kahan-Babuška-Neumaier summation) checks the relative magnitudes of the accumulator and the new summand to ensure the lost bits are correctly captured regardless of which is larger.

## Task
Implement the `compensated_sum` function that computes the sum of a 1D `float32` array using the Kahan-Neumaier algorithm. 

The algorithm maintains an accumulator $s$ and a compensation $c$, both initialized to `0.0` (as `float32`). For each element $x$ in the array:
1. Compute the tentative sum $t = s + x$.
2. Determine which of $s$ and $x$ is larger in absolute value.
3. If $|s| \geq |x|$, the lost low-order bits are $(s - t) + x$.
4. If $|s| < |x|$, the lost low-order bits are $(x - t) + s$.
5. Add these lost bits to the compensation $c$.
6. Update $s = t$.
Finally, return $s + c$.

All intermediate arithmetic operations MUST be performed in `float32`. Do not simply cast to `float64`. Use `np.float32()` to enforce single precision.

## Example
```python
import numpy as np

arr = np.array([1e13, 1.0, -1e13], dtype=np.float32)

# Naive sum (evaluated left-to-right) yields 0.0
# (1e13 + 1.0) -> 1e13
# 1e13 - 1e13 -> 0.0

# Compensated sum yields 1.0
ans = compensated_sum(arr)
```

## What the gate checks
The gate evaluates your function on adversarial `float32` arrays designed to induce catastrophic cancellation in standard summation. It compares your output's relative error (`rel_err`) against a ground-truth `float64` sum. A naive `float32` loop or `np.sum(arr, dtype=np.float32)` will fail the `rel_err <= 1e-6` threshold, but a correctly implemented Kahan-Neumaier sum will pass.
