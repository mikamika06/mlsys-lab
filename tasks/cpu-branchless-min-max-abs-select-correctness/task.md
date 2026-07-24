## Context

When a CPU executes an `if` statement it must predict which branch will be taken. A wrong prediction causes a pipeline flush and a noticeable penalty. *Branch‑less* code avoids this by using arithmetic or bitwise tricks instead of conditional jumps.

A common pattern is to compute element‑wise operations on integer arrays without explicit branching, e.g.
$$
\min(a,b) = \frac{a+b- |a-b|}{2}, \qquad
\max(a,b) = \frac{a+b+ |a-b|}{2}
$$
or using vectorised NumPy functions that internally use SIMD instructions.

In this task you will implement a small helper that returns several such branch‑free results and verify them against the reference implementation that uses the safe, but branchy, `np.minimum`, `np.maximum` and `np.where`.

## Task

Implement the function

```python
def branchless_ops(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    ...
```

It receives two integer arrays `x` and `y` of equal shape and a boolean mask array `mask`.  The function must return a 5‑tuple containing

1. element‑wise minimum of `x` and `y`
2. element‑wise maximum of `x` and `y`
3. absolute value of `x`
4. absolute value of `y`
5. an array that selects elements from `x` where `mask` is true, otherwise from `y`

All operations must be performed without using Python level branching (no `if`, no `np.where` with a Python function).  The result should use NumPy vectorised operations and produce arrays of the same dtype as the inputs.

## Example

```python
import numpy as np
x = np.array([3, -4, 7], dtype=np.int32)
y = np.array([5, 2, -6], dtype=np.int32)
mask = np.array([True, False, True])

mins, maxs, abs_x, abs_y, sel = branchless_ops(x, y, mask)

print(mins)   # [3, -4, -6]
print(maxs)   # [5, 2, 7]
print(abs_x)  # [3, 4, 7]
print(abs_y)  # [5, 2, 6]
print(sel)    # [3, 2, 7]
```

## What the gate checks

The grader generates several random test cases and compares your output arrays against a reference implementation that uses `np.minimum`, `np.maximum`, `np.abs` and `np.where`.  It computes the byte‑wise fraction of identical bytes between your results and the reference.  The submission passes only if this value equals **1.0** for all test cases, i.e. every output element matches exactly.
