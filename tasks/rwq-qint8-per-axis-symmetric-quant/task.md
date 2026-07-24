## Context

Per-tensor int8 quantization gives a weight matrix a single scale, so
one loud output channel forces every other channel to share its (too
coarse) resolution. Production quantizers instead use **per-axis
(per-channel) symmetric** int8: each output channel gets its own scale,
derived from that channel's own absmax, with **no zero-point** (weights
are typically zero-centered, so a symmetric range wastes nothing). This
is exactly what PyTorch's `torch.quantize_per_channel` and TensorFlow
Lite's per-channel weight quantization do for conv/linear weights.

### Formula

For a weight tensor $W$ and a channel axis $a$, let $W_c$ denote the
slice of $W$ at index $c$ along axis $a$ (all other axes free). For each
channel $c$:
$$
\mathrm{absmax}_c = \max |W_c| \qquad (\text{use } 1 \text{ if this is } 0)
$$
$$
s_c = \frac{\mathrm{absmax}_c}{127}
\qquad
q = \mathrm{clip}(\mathrm{round}(W_c / s_c),\ -127,\ 127)
\qquad
\widehat{W}_c = q \cdot s_c
$$
Codes are restricted to $[-127, 127]$ (not $-128$) so the range is
perfectly symmetric around zero — the standard convention for symmetric
qint8.

## Task

Implement:

```python
def per_axis_qint8(W: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `W` — 2-D weight matrix.
* `axis` — the channel axis: one scale per index along this axis (e.g.
  `axis=0` on a `(d_out, d_in)` matrix means one scale per output row).

Return `(codes, scale, dequant)`:

* `codes` — integer array, same shape as `W`, values in $[-127, 127]$.
* `scale` — array broadcastable against `W`, one value per channel (shape
  has size 1 on every axis except `axis`).
* `dequant` — `codes * scale`, same shape as `W`.

## Example

```python
import numpy as np
W = np.array([[1.0, -2.0, 4.0],
              [0.5,  0.5, -0.5]])
codes, scale, deq = per_axis_qint8(W, axis=0)
# row 0: absmax=4 -> scale=4/127; row 1: absmax=0.5 -> scale=0.5/127
# each row quantized independently at its own resolution
```

## What the gate checks

* **exact_match** — your `codes` must equal, element for element, a
  NumPy oracle computing per-axis absmax/127 symmetric codes as above, on
  several random matrices (both `axis=0` and `axis=1`, including a
  channel forced to all-zero to exercise the absmax-guard).
* **max_abs_err** — the maximum absolute difference between your
  `dequant` and the oracle's must be $\le 10^{-6}$ on the same cases.
