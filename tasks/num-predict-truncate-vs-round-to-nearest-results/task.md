## Context

When a Python `float` (which is a C double) is cast to a NumPy `float32`, the value is rounded according to the current IEEE‑754 rounding mode. The default mode in NumPy is *round‑to‑nearest‑even*; that means the nearest representable binary number is chosen, and if the target lies exactly halfway between two candidates the one with an even least significant bit is selected.

An alternative mode is *truncation*, i.e. rounding toward zero. In this mode the fractional part of the mantissa is simply discarded, so the result is always closer to zero than or equal in magnitude to the original value.

The task is to predict, for a given array of `float64` values, what the resulting `float32` array would be under each of these two rounding modes.

## Task

Implement `predict_rounding_results(arr, mode)`:

```python
def predict_rounding_results(arr: np.ndarray, mode: str) -> np.ndarray:
    ...
```

* `arr` is a one‑dimensional NumPy array of dtype `float64`.
* `mode` must be either `"nearest"` (default rounding) or `"trunc"` (round toward zero).
* The function should return a new array of the same shape, with dtype `float32`, containing the values that would result from converting each element of `arr` to `float32` using the specified rounding mode.

The implementation must use only NumPy vectorised operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
A = np.array([1.5, -2.3, 0.7], dtype=np.float64)

nearest = predict_rounding_results(A, "nearest")
# array([ 1.5, -2.3,  0.7], dtype=float32)

trunc = predict_rounding_results(A, "trunc")
# array([ 1., -2.,  0.], dtype=float32)
```

## What the gate checks

The grader computes a reference result using NumPy’s own conversion for `"nearest"` and `np.trunc` followed by a cast to `float32` for `"trunc"`.  
It then compares the bit‑patterns of your output with those of the reference using an exact match.  The solution must therefore produce identical `float32` values, not just numerically close ones.

The gate metric is `exact_match`; it succeeds only if all test cases pass exactly.
