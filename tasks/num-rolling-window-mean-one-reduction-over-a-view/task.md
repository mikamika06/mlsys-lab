## Context

A rolling window over a one-dimensional signal groups consecutive values without
creating a Python object for every window. A zero-copy view can expose these
overlapping windows, and a reduction over the last axis computes each window's
summary.

For a signal $x \in \mathbb{R}^n$ and window size $w$, the rolling mean output
$y \in \mathbb{R}^{n-w+1}$ is

$$
y_i = \frac{1}{w}\sum_{j=0}^{w-1} x_{i+j}.
$$

A view-based implementation can represent the windows as a matrix
$V \in \mathbb{R}^{(n-w+1)\times w}$ where

$$
V_{ij}=x_{i+j}.
$$

The result is then a single vectorized reduction:

$$
y = \frac{\operatorname{sum}(V,\mathrm{axis}=1)}{w}.
$$

NumPy's stride machinery allows constructing such overlapping views without
copying the input buffer. The important distinction is that reshaping or stacking
temporary arrays may allocate memory, while a view keeps the original storage.

## Task

Implement `rolling_window_mean(x, window)`:

```python
def rolling_window_mean(x: np.ndarray, window: int) -> np.ndarray:
    ...
```

The function takes a one-dimensional NumPy array `x` and a positive integer
`window`. It returns a `float64` NumPy array containing the mean of every
contiguous window of length `window`.

Use a zero-copy window view internally with NumPy stride operations and perform
one vectorized reduction over the window dimension. Do not use Python loops over
the windows.

Assume `window <= len(x)`.

## Example

```python
import numpy as np

x = np.array([1., 2., 3., 4., 5.])
y = rolling_window_mean(x, 3)

# [2., 3., 4.]
```

## What the gate checks

The gate computes a reference result using NumPy's own sliding-window view
implementation and compares the candidate output with the reference using the
relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}}-y_{\mathrm{reference}}\rVert_2}
{\lVert y_{\mathrm{reference}}\rVert_2+10^{-12}}.
$$

The submitted implementation must satisfy
$\mathrm{rel\_err} \le 10^{-12}$ on several deterministic inputs. The grader
also traces Python execution to discourage implementations that iterate over
individual windows.
