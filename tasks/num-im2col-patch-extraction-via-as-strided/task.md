## Context

Convolution can be turned into a single matrix multiply once every sliding
window ("patch") of the input has been laid out as its own row or slice —
the classic **im2col** trick. The naive way to build that patch tensor
copies every window into fresh memory. But a sliding window over a 2-D array
is really just the *same bytes*, addressed with different strides: instead
of copying, `numpy.lib.stride_tricks.as_strided` can construct a **view**
that reads those overlapping windows directly out of the original buffer.

For an input $x \in \mathbb{R}^{H \times W}$ with row/column strides
$(s_0, s_1)$ (in bytes), a patch of size $(k_h, k_w)$ extracted with stride
$t$ starting at output position $(i, j)$ covers

$$
\text{patches}[i, j, a, b] = x[i \cdot t + a,\; j \cdot t + b],
\qquad 0 \le a < k_h,\; 0 \le b < k_w .
$$

The full patch tensor has shape $(\text{out}_h, \text{out}_w, k_h, k_w)$
where

$$
\text{out}_h = \left\lfloor \frac{H - k_h}{t} \right\rfloor + 1, \qquad
\text{out}_w = \left\lfloor \frac{W - k_w}{t} \right\rfloor + 1 .
$$

This whole tensor can be described with exactly four strides —
$(t \cdot s_0,\; t \cdot s_1,\; s_0,\; s_1)$ — over the *same* underlying
buffer as $x$, with no copy of any element, even though neighbouring
patches overlap and reuse the same bytes multiple times.

## Task

Implement `im2col_patches`:

```python
def im2col_patches(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    ...
```

- `x` is a 2-D `float64` NumPy array of shape `(H, W)`, C-contiguous.
- `kh`, `kw` are the patch height and width (`kh <= H`, `kw <= W`).
- `stride` is the step between successive patches (`stride >= 1`), the same
  in both dimensions.
- Return an array of shape `(out_h, out_w, kh, kw)` as defined above,
  containing the extracted patches.
- The result must be a **zero-copy view** onto `x`'s own memory buffer —
  build it with explicit strides (e.g. via
  `numpy.lib.stride_tricks.as_strided`), not by copying data into a new
  array (no Python loops appending patches, no `np.stack`/`np.array` of a
  patch list, no `.copy()`).

## Example

```python
import numpy as np

x = np.arange(16, dtype=np.float64).reshape(4, 4)
patches = im2col_patches(x, kh=2, kw=2, stride=2)

# patches.shape == (2, 2, 2, 2)
# patches[0, 0] == [[0., 1.], [4., 5.]]
# patches[0, 1] == [[2., 3.], [6., 7.]]
# patches[1, 0] == [[8., 9.], [12., 13.]]
```

## What the gate checks

The grader draws several random `(H, W, kh, kw, stride)` combinations and
builds the reference patch tensor with NumPy's own
`numpy.lib.stride_tricks.sliding_window_view` (a real, independent oracle),
sub-sampled by `stride`. Two gates apply to every case:

- **`byte_exact_fraction`** — your output must match the oracle's values
  exactly, byte for byte.
- **`is_view`** — `numpy.shares_memory(your_output, x)` must be `True`,
  confirming the result is a genuine zero-copy view over `x`'s buffer and
  not a freshly allocated copy.

Both gates must hold on the worst case across all generated inputs.
