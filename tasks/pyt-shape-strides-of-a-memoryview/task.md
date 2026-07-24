## Context

NumPy arrays store their data as a contiguous (or strided) block of bytes
behind the scenes.  The **buffer protocol** lets any Python object expose that
raw memory without copying.  A `memoryview` wraps a buffer-producing object and
exposes five key properties that describe how multi-dimensional indices map to
byte offsets:

$$\text{offset}(i_0, i_1, \ldots, i_{d-1})
  = \sum_{k=0}^{d-1} i_k \cdot s_k$$

where $s_k$ is the **stride** (in bytes) along axis $k$.  For a
C-contiguous array with shape $(n_0, n_1, \dots, n_{d-1})$ and element size
$t$, the strides are

$$s_k = t \cdot \prod_{j=k+1}^{d-1} n_j$$

so that walking along the *last* axis advances by $t$ bytes, along the
second-to-last by $n_{d-1} \cdot t$ bytes, and so on.

Transposing, slicing with a step, or choosing Fortran (`'F'`) order all
change the strides while keeping the same underlying data buffer.  The
`memoryview` object faithfully reflects whichever layout the array actually
uses.

Five properties are available on every `memoryview`:

| Property | Meaning |
|---|---|
| `shape` | tuple of lengths along each axis |
| `strides` | tuple of byte-steps along each axis |
| `itemsize` | bytes per element |
| `ndim` | number of axes |
| `format` | struct format character (e.g. `'d'` for float64, `'f'` for float32, `'l'` for int64) |

## Task

Implement `memoryview_info`:

```python
def memoryview_info(arr: np.ndarray) -> dict:
    ...
```

It takes a NumPy array and returns a `dict` with exactly five keys:
`"shape"`, `"strides"`, `"itemsize"`, `"ndim"`, `"format"` — the values
from the array's `memoryview`.

You **must** use the built-in `memoryview` to obtain these values (not
`arr.shape` or `arr.strides` directly).  The returned `shape` and `strides`
must be **tuples of ints** (not lists or numpy arrays).

## Example

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
info = memoryview_info(arr)
# {'shape': (2, 3), 'strides': (24, 8), 'itemsize': 8, 'ndim': 2, 'format': 'd'}
```

A transposed array changes the strides:

```python
info_t = memoryview_info(arr.T)
# {'shape': (3, 2), 'strides': (8, 24), 'itemsize': 8, 'ndim': 2, 'format': 'd'}
```

## What the gate checks

The gate creates twelve NumPy arrays spanning different dtypes (`int32`,
`int64`, `float16`, `float32`, `float64`, `uint8`), dimensionalities (1-D,
2-D, 3-D), memory layouts (C-contiguous, Fortran-ordered, transposed, strided
slices, non-contiguous sub-views), and even an empty array.  For each it
builds a real `memoryview` and compares every key of your dict against the
oracle.  Any mismatch — including wrong types (e.g. list instead of tuple) —
scores zero.
