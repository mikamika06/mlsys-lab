## Context

Given two arrays $a$ and $b$, NumPy's broadcasting rule expands both to a
common shape without copying data. Right-align the two shape tuples,
left-padding the shorter one with $1$s. For each aligned pair of dimensions
$(d_a, d_b)$, the output dimension is
$$
d_{\text{out}} =
\begin{cases}
d_a, & d_a = d_b, \\
d_b, & d_a = 1, \\
d_a, & d_b = 1, \\
\text{error}, & \text{otherwise.}
\end{cases}
$$
Whenever a source dimension is expanded from $1$ to $d_{\text{out}}>1$, the
corresponding output stride is set to $0$: every step along that axis reads
the same memory address, so no bytes are ever copied — this is exactly how
`np.broadcast_arrays(a, b)` works internally.

## Task

Implement `broadcast_pair`:

```python
def broadcast_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Compute the common output shape by hand using the rule above (do not call
`np.broadcast_shapes`), then build **zero-copy views** of `a` and `b` at that
shape with `numpy.lib.stride_tricks.as_strided` — set `0` as the stride on
every axis that had to be expanded from size `1`. Raise `ValueError` if the
two shapes are not broadcast-compatible. `np.broadcast_arrays`,
`np.broadcast_to`, `np.broadcast_shapes`, `np.tile`, `np.repeat` and
`np.resize` are blocked by the grader — build the strides yourself.

## Example

```python
import numpy as np
a = np.arange(3.0)              # shape (3,)
b = np.ones((4, 3))             # shape (4, 3)
va, vb = broadcast_pair(a, b)
va.shape, va.strides            # (4, 3), (0, 8)
np.shares_memory(va, a)         # True -- no copy
```

## What the gate checks

Eight pairs of arrays (scalars, leading-axis padding, size-1 middle axes, a
non-contiguous source, an `int32` source, and an identity case) are compared
against the real `np.broadcast_arrays(a, b)` oracle:

* `byte_exact_fraction` — mean fraction of matching bytes between your two
  returned arrays and the oracle's two arrays, materialised and compared
  element-for-element; must be `1.0`.
* `zero_copy_fraction` — fraction of returned arrays that share memory with
  their source (`np.shares_memory`) and have exactly the oracle's `strides`;
  must be `1.0`. Materialising a copy of the data fails here even when the
  values are correct.
