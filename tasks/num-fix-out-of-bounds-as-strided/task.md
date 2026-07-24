## Context

NumPy arrays store elements in a contiguous or strided memory layout. A view can
reuse the same memory without copying by describing a new shape and set of
strides. The function `numpy.lib.stride_tricks.as_strided` can create such views,
but it does not check whether the requested view stays inside the original
buffer.

For a one-dimensional array $x$ with length $n$ and element stride $s$, a sliding
window view of width $w$ has

$$
m = n - w + 1
$$

rows. The view shape is $(m, w)$ and both dimensions advance using the original
stride:

$$
\mathrm{shape} = (m, w), \qquad
\mathrm{strides} = (s, s).
$$

The last element accessed is at offset

$$
(m - 1)s + (w - 1)s = (n - 1)s,
$$

so the view remains within the original array.

## Task

Implement `fixed_windows(x, width)`:

```python
def fixed_windows(x: np.ndarray, width: int) -> np.ndarray:
    ...
```

Return a zero-copy two-dimensional NumPy view containing every consecutive
window of `width` elements from the one-dimensional input array `x`.

Requirements:

- The returned array must have shape `(len(x) - width + 1, width)`.
- The returned array must use the same underlying storage as `x`.
- Use `numpy.lib.stride_tricks.as_strided` or equivalent stride manipulation.
- Do not create a copied array with `reshape`, `stack`, or explicit indexing.
- The returned view must not access memory outside `x`.

## Example

```python
import numpy as np

x = np.array([10, 20, 30, 40], dtype=np.int64)

w = fixed_windows(x, 3)

# w is a view with:
# [[10, 20, 30],
#  [20, 30, 40]]
```

## What the gate checks

The gate constructs valid inputs and computes the expected window view using
NumPy's stride machinery with a safe shape and stride configuration.

The returned bytes must match the NumPy reference exactly. The gate also rejects
views whose memory range extends beyond the original array buffer, because an
invalid `as_strided` configuration can appear to work while reading unrelated
memory.
