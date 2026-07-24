## Context

An inclusive scan transforms a sequence into cumulative results. For an input
array $x = [x_0, x_1, \dots, x_{n-1}]$, the output $y$ is defined by

$$
y_i = \sum_{j=0}^{i} x_j .
$$

Each output element includes the current input element, which distinguishes an
inclusive scan from an exclusive scan. An exclusive scan would place the sum of
previous elements at each position instead.

A prefix sum is the common example of an inclusive scan. For example, the
prefix sum of $[2, 3, 5]$ is $[2, 5, 10]$ because

$$
[2,\ 2+3,\ 2+3+5] = [2,\ 5,\ 10].
$$

## Task

Implement `inclusive_scan(x)`:

```python
def inclusive_scan(x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a one-dimensional NumPy array of numeric values and must
return a one-dimensional NumPy array containing the inclusive prefix sums.

The output must have the same shape as the input and use `float64` values.
Compute the scan in order from the first element to the last.

## Example

```python
import numpy as np

x = np.array([2, 3, 5])
y = inclusive_scan(x)

# y is:
# [ 2.  5. 10.]
```

## What the gate checks

The gate compares the returned values against a NumPy `np.cumsum` reference
implementation. The maximum absolute difference

$$
\max_i |y_i - y_i^{\mathrm{ref}}|
$$

must be less than $10^{-9}$.
