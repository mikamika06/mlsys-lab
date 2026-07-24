## Context

Every NumPy array is really just a flat block of bytes plus a **shape**, a
**strides** tuple and a byte **offset** that together describe how to read
logical elements out of that block. For a `dtype` of itemsize $s$ (8 for
`float64`), a multi-index $\mathbf{i} = (i_0, i_1, \dots, i_{k-1})$ into an
array of shape $(n_0, \dots, n_{k-1})$ and strides $(\sigma_0, \dots,
\sigma_{k-1})$ (strides measured in **bytes**) maps to a flat element position

$$
p(\mathbf{i}) \;=\; \frac{\text{offset}\cdot s \;+\; \sum_{d=0}^{k-1} i_d\, \sigma_d}{s},
$$

where $\text{offset}$ is measured in **elements**. This single formula is how
reshapes, transposes, slices and broadcasts are all implemented without ever
copying data: only $(shape, strides, offset)$ change, never the underlying
buffer. `numpy.lib.stride_tricks.as_strided` builds exactly this kind of view
directly from a shape/strides pair, which is what makes it the natural oracle
for checking a from-scratch implementation of the same index math.

## Task

Implement:

```python
def flat_gather(buf: np.ndarray, shape: tuple, strides: tuple, offset: int) -> np.ndarray:
    ...
```

* `buf` — a 1-D `float64` NumPy array: the flat backing buffer.
* `shape` — the logical shape of the array to materialize.
* `strides` — a tuple of **byte** strides, one per axis of `shape` (the same
  convention as `ndarray.strides`; strides may be zero, for a broadcast axis,
  or negative, for a reversed axis).
* `offset` — the starting position into `buf`, in **elements** (not bytes).

Return a **new** `float64` array of shape `shape` whose entry at multi-index
$\mathbf{i}$ is `buf.ravel()[p(i)]` using the formula above. Do not use
`numpy.lib.stride_tricks.as_strided` (or anything that reads the buffer
through its native strides machinery) — compute each flat position yourself
from `shape`, `strides` and `offset`.

## Example

```python
import numpy as np

buf = np.arange(24, dtype=np.float64)     # itemsize = 8 bytes

# Plain reshape to (4, 6): row stride = 6*8 = 48 bytes, col stride = 8 bytes
out = flat_gather(buf, shape=(4, 6), strides=(48, 8), offset=0)
assert np.array_equal(out, buf.reshape(4, 6))

# Transpose of that same (4, 6) block: swap the strides
outT = flat_gather(buf, shape=(6, 4), strides=(8, 48), offset=0)
assert np.array_equal(outT, buf.reshape(4, 6).T)

# Broadcast a 5-element run across 4 rows: row stride = 0
bcast = flat_gather(buf, shape=(4, 5), strides=(0, 8), offset=3)
assert np.array_equal(bcast, np.tile(buf[3:8], (4, 1)))
```

## What the gate checks

For several `(buf, shape, strides, offset)` cases — including a plain reshape,
a transpose (swapped strides), an offset sub-block, a zero-stride broadcast,
and a negative-stride reversal — the grader builds the true view with
`numpy.lib.stride_tricks.as_strided(buf[offset:], shape=shape, strides=strides)`
and copies it out as the reference. It compares your output to that reference
byte-for-byte with `byte_exact_fraction` (shape mismatches score `0.0`) and
takes the minimum across all cases. The gate requires this minimum to be
`>= 1.0`, i.e. every case must match exactly.
