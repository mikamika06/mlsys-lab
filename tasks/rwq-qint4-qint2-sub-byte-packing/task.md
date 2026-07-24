## Context

Low-bit quantization only pays off if the codes are actually stored at
that bit width — a "4-bit" scheme that keeps one `int8` per code wastes
half its promised savings. Production quantized-model formats **pack**
multiple sub-byte codes into each byte. This task implements that packing
(and its inverse, unpacking for dequantization) for two widths:
**qint4** (2 codes per byte) and **qint2** (4 codes per byte).

### Quantize

Per row, symmetric quantization with $q_{\max} = 2^{b-1}-1$ ($b$ = `nbits`):

$$
s = \frac{\max_j |W_{\cdot,j}|}{q_{\max}} \quad (\text{or } 1 \text{ if the row is all zero})
$$
$$
\text{code}_j = \mathrm{clip}(\mathrm{round}(W_{\cdot,j}/s),\, -q_{\max},\, q_{\max}), \qquad
u_j = \text{code}_j + q_{\max}
$$

$u_j$ is unsigned and fits in $b$ bits: $u_j \in [0,\, 2q_{\max}]$.

### Pack

With `per_byte = 8 // nbits` codes per byte, pack **least-significant code
first** (code $0$ occupies the low `nbits` bits of the byte, code $1$ the
next `nbits` bits, and so on):

$$
\text{byte}_k = \sum_{t=0}^{\text{per\_byte}-1} u_{k\cdot\text{per\_byte}+t} \cdot 2^{t\cdot b}
$$

`d_in` is guaranteed divisible by `per_byte`, so each byte holds exactly
`per_byte` whole codes and the packed row has `d_in * nbits // 8` bytes.

### Unpack / dequantize

Reverse the bit-shifting to recover each $u_j$ (mask `nbits` bits at
offset $t\cdot b$), then $\hat W_j = (u_j - q_{\max})\cdot s$.

## Task

Implement `pack_sub_byte`:

```python
def pack_sub_byte(W: np.ndarray, nbits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `nbits` — `4` or `2`; `d_in` is divisible by `8 // nbits`.

Return `(packed, s, dequant)`:

* `packed` — `uint8` array, shape `(d_out, d_in * nbits // 8)`, the packed
  buffer above.
* `s` — float array, shape `(d_out,)`, the per-row scale.
* `dequant` — float array, shape `(d_out, d_in)`, the reconstruction
  obtained by unpacking `packed` (not by skipping straight from `code`).

## Example

```python
import numpy as np
W = np.array([[1.0, -2.0, 0.5, 2.0]])  # d_in=4
packed4, s4, deq4 = pack_sub_byte(W, nbits=4)  # 2 codes/byte -> 2 bytes
packed2, s2, deq2 = pack_sub_byte(W, nbits=2)  # 4 codes/byte -> 1 byte
```

## What the gate checks

* **byte_exact_fraction** — your `packed` buffer, for both `nbits=4` and
  `nbits=2`, must be **byte-identical** to a NumPy oracle running the
  quantize+pack recipe above on the fixed weight fixture (`qnt_w.npy`);
  the reported value is the worst of the two trials.
* **dequant_max_abs_err** — your `dequant` must match the oracle's
  unpack-then-reconstruct output (max abs error) for both bit widths; the
  reported value is the worst of the two trials.
