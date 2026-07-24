## Context

Low precision floating point formats reduce memory usage but also reduce the
precision available during arithmetic. In particular, repeatedly adding values
in `float16` can accumulate large rounding errors because each intermediate
sum is stored in a small precision format.

For a row vector $x \in \mathbb{R}^d$, the reduction

$$s = \sum_{i=1}^{d} x_i$$

should preserve as much information as possible during accumulation. A common
mixed-precision strategy is to store inputs in `float16` while performing
reductions in `float32` or higher precision.

NumPy reductions can control the accumulator type through the `dtype` argument.
The reference computation uses a higher precision accumulator:

$$s_{\mathrm{ref}} = \sum_{i=1}^{d} \mathrm{float64}(x_i).$$

The goal is not to change the input representation, but to avoid losing
accuracy while computing the reduction.

## Task

Implement `sum_rows_fp32(A)`:

```python
def sum_rows_fp32(A: np.ndarray) -> np.ndarray:
    ...
```

The input is a 2-D NumPy array containing `float16` values with shape
$(n, d)$. Return a 1-D NumPy array of length $n$ containing the sum of each row.

The accumulation must happen in `float32` or better precision. Do not convert
the input data permanently to another storage format. The returned array should
have dtype `float32`.

## Example

```python
import numpy as np

A = np.array(
    [
        [1000, 1, -1000, 1],
        [0.5, 0.5, 0.5, 0.5],
    ],
    dtype=np.float16,
)

out = sum_rows_fp32(A)
# array([2., 2.], dtype=float32)
```

## What the gate checks

The gate compares the implementation against a NumPy oracle that computes the
same row sums using `float64` accumulation and converts the result to
`float32`. The relative error

$$\mathrm{rel\_err} =
\frac{\lVert y - y_{\mathrm{ref}}\rVert_2}
{\lVert y_{\mathrm{ref}}\rVert_2 + 10^{-12}}$$

must satisfy $\mathrm{rel\_err} \le 10^{-4}$.

A solution that accumulates directly in `float16` loses too much information on
rows with many additions and fails the accuracy requirement.
