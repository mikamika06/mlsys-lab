## Context

GGML's `Q4_K` format packs a 256-value super-block into 8 sub-blocks of 32
values. Each sub-block has its own 6-bit scale $s_j$ and 6-bit minimum $m_j$
($j = 0..7$), which are dequantized as

$$
w_i = d \cdot s_j \cdot q_i - d_{\min} \cdot m_j ,
$$

with $d$ and $d_{\min}$ the two `float16` super-block scales. Sixteen 6-bit
numbers ($8$ scales $+$ $8$ mins $= 96$ bits) are squeezed into exactly
**12 bytes**, but not with a simple bit-stream: `ggml`'s `get_scale_min_k4`
uses an asymmetric layout tuned so the low 4 sub-blocks each fit in one byte,
and the high 4 sub-blocks borrow their missing 2 bits from spare bits left
over in the first two groups of four bytes.

Byte layout of the 12-byte blob `q[0..11]`, in terms of the 6-bit values
$s_j$ (scale) and $m_j$ (min):

* `q[0..3]`: low 6 bits $= s_{0..3}$, high 2 bits $=$ high 2 bits of
  $s_{4..7}$ (one sub-block's high bits per byte).
* `q[4..7]`: low 6 bits $= m_{0..3}$, high 2 bits $=$ high 2 bits of
  $m_{4..7}$.
* `q[8..11]`: low 4 bits $=$ low 4 bits of $s_{4..7}$, high 4 bits $=$ low 4
  bits of $m_{4..7}$.

This is exactly the bit-packing implemented by `get_scale_min_k4` in
`ggml`'s dequantization code, reproduced here in C:

```c
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4);
    }
}
```

## Task

Implement `unpack_q4k_scales_mins` in `solve.py`:

```python
def unpack_q4k_scales_mins(packed: np.ndarray):
    ...
```

* `packed` — `uint8` array of shape `(12,)`, the raw scales/mins blob.

Return a tuple `(scales, mins)`:

* `scales` — `uint8` array of shape `(8,)`, the 8 unpacked 6-bit scale codes
  $s_0 .. s_7$, each in `[0, 63]`.
* `mins` — `uint8` array of shape `(8,)`, the 8 unpacked 6-bit min codes
  $m_0 .. m_7$, each in `[0, 63]`.

Reproduce `get_scale_min_k4` for every $j = 0..7$: the first 4 sub-blocks
read straight from a single byte each, the last 4 must be reassembled from
4 low bits in `q[8..11]` plus 2 high bits borrowed from `q[0..7]`.

## Example

```python
import numpy as np

packed = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.uint8)
scales, mins = unpack_q4k_scales_mins(packed)
# scales[j] and mins[j] each in [0, 63], reconstructed per get_scale_min_k4
```

## What the gate checks

The grader ports `get_scale_min_k4` to NumPy/Python and runs it as the
oracle on a mix of blobs: an all-zero blob, an all-`63` (max) blob, several
blobs built by packing known random 6-bit scale/min vectors (so the round
trip is verifiable end to end), and several raw random 12-byte blobs (so the
unpacker must match the oracle's decoding bit-for-bit, not just on
"nice" inputs). All 8 scales and 8 mins must match the oracle exactly —
`exact_match` requires every value across every scenario to be identical.
