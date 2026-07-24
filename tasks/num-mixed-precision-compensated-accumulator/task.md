## Context

Floating point addition is not exact. When many values with different magnitudes are accumulated, small terms can be lost because the running total has limited precision.

For a sequence of values $x_1, x_2, \dots, x_n$, the exact sum is

$$
S = \sum_{i=1}^{n} x_i .
$$

A mixed-precision accumulator can use low precision storage and higher precision arithmetic. In this task, inputs are stored as `float16`, while accumulation must happen in `float32`.

Kahan compensated summation keeps an additional correction value that tracks lost low-order bits. The update for each value $x$ is

$$
y = x - c,
$$

$$
t = s + y,
$$

$$
c = (t - s) - y,
$$

where $s$ is the accumulated sum and $c$ is the compensation term.

The final result should approximate a high precision reference computed in `float64`.

## Task

Implement `compensated_sum(x)`:

```python
def compensated_sum(x):
    ...
```

The function receives a one-dimensional NumPy array with dtype `float16`. Return a Python `float` containing the sum using a compensated accumulation in `float32`.

Do not convert the entire input to `float64` before summing. The purpose of the task is to preserve accuracy while accumulating in `float32`.

## Example

```python
import numpy as np

x = np.array([1e4, 1.0, -1e4, 3.0], dtype=np.float16)
result = compensated_sum(x)
```

The returned value should be close to the `float64` reference sum of the same `float16` values.

## What the gate checks

The gate creates adversarial `float16` arrays containing values with large cancellation and many small contributions. It computes the oracle result using NumPy `float64` summation and measures

$$
\mathrm{rel\_err} =
\frac{\lVert y - y_{\mathrm{ref}} \rVert}
{\lVert y_{\mathrm{ref}} \rVert + 10^{-12}} .
$$

The returned value must have $\mathrm{rel\_err} < 10^{-4}$. A plain `float32` accumulator without compensation loses enough information on the generated cases and does not pass.
