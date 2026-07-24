## Context

`Q6_K` is a "K-quant" weight format used by `ggml`/`llama.cpp`. It packs a
**super-block** of $256$ values into:

- one `fp16` super-scale $d$,
- $16$ signed 8-bit sub-scales $\mathrm{sc}_0, \dots, \mathrm{sc}_{15}$ (one per
  group of $16$ values),
- $128$ bytes `ql` holding the low 4 bits of each 6-bit code,
- $64$ bytes `qh` holding the high 2 bits of each 6-bit code.

Each of the 256 values is quantized to a 6-bit signed code $q \in [-32, 31]$
(stored as an unsigned value in $[0, 63]$, biased by $32$), and reconstructed as

$$
\hat{x} = d \cdot \mathrm{sc}_{\,\mathrm{is}} \cdot q .
$$

The packing is organized in two halves of $128$ output values each
(`ql`/`qh` split into two 64/32-byte halves). Within a half, for
$l = 0, \dots, 31$, four 6-bit codes are recovered from `ql[l]`, `ql[l+32]`,
and `qh[l]`:

$$
\begin{aligned}
q_1 &= \big((\mathrm{ql}[l] \;\&\; 0\mathrm{xF}) \;|\; ((\mathrm{qh}[l] \gg 0 \;\&\; 3) \ll 4)\big) - 32 \\
q_2 &= \big((\mathrm{ql}[l+32] \;\&\; 0\mathrm{xF}) \;|\; ((\mathrm{qh}[l] \gg 2 \;\&\; 3) \ll 4)\big) - 32 \\
q_3 &= \big((\mathrm{ql}[l] \gg 4) \;|\; ((\mathrm{qh}[l] \gg 4 \;\&\; 3) \ll 4)\big) - 32 \\
q_4 &= \big((\mathrm{ql}[l+32] \gg 4) \;|\; ((\mathrm{qh}[l] \gg 6 \;\&\; 3) \ll 4)\big) - 32
\end{aligned}
$$

with $\mathrm{is} = \lfloor l / 16 \rfloor$ selecting a pair of sub-scale
indices that shift by $2$ between $q_1, q_2, q_3, q_4$ (offsets $0, 2, 4, 6$
from the half's sub-scale base). $q_1$ lands at output offset $l$, $q_2$ at
$l+32$, $q_3$ at $l+64$, $q_4$ at $l+96$ (relative to the half's base offset,
$0$ or $128$).

## Task

Implement `q6_k_dequantize(d, scales, ql, qh)`:

```python
def q6_k_dequantize(d: float, scales: np.ndarray, ql: np.ndarray, qh: np.ndarray) -> np.ndarray:
    ...
```

Arguments:

- `d`: a Python float / NumPy scalar — the super-block scale (already
  converted from `fp16` to a plain float; do not re-quantize it).
- `scales`: array-like of `16` signed 8-bit sub-scales (`int8`, values in
  $[-128, 127]$).
- `ql`: array-like of `128` bytes (`uint8`) — low nibbles of the 6-bit codes.
- `qh`: array-like of `64` bytes (`uint8`) — high 2-bit pairs of the 6-bit
  codes.

Return a NumPy array of shape `(256,)` and dtype `float32` holding the
reconstructed values $\hat{x}_0, \dots, \hat{x}_{255}$ following the equations
above. Do not change the function name or argument order.

## Example

```python
import numpy as np

d = 0.02
scales = np.zeros(16, dtype=np.int8)
scales[:] = 10
ql = np.zeros(128, dtype=np.uint8)
qh = np.zeros(64, dtype=np.uint8)

# ql[l] == 0, qh[l] == 0 for all l -> every 6-bit code is 0 - 32 = -32
x = q6_k_dequantize(d, scales, ql, qh)
# x[j] == 0.02 * 10 * (-32) == -6.4 for every j
```

## What the gate checks

The gate builds an independent NumPy oracle that ports `dequantize_row_q6_K`
from `ggml` (looping over the two 128-value halves and the packed nibble/bit
layout described above), feeds it the same randomly generated `d`, `scales`,
`ql`, `qh`, and compares the oracle's reconstruction against the submission's
elementwise. The submission passes when the maximum absolute error over all
$256$ values is at most $10^{-6}$.
