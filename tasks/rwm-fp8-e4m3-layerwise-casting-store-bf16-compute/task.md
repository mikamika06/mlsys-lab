## Context

"Layerwise casting" (used e.g. by diffusers/Transformer-Engine style fp8
storage) keeps a layer's weights **at rest** in the tiny 8-bit `E4M3` float
format, and **upcasts** them just-in-time to `bfloat16` precision for the
actual matmul, so compute uses a well-behaved 16-bit grid while storage stays
at 1 byte/parameter.

### The E4M3 format (OCP FP8, "E4M3FN" — finite, no infinities)

A code is 8 bits: `[sign(1) | exponent(4) | mantissa(3)]`, bias $= 7$.

$$
\text{value}(s, e, m) =
\begin{cases}
(-1)^s \cdot 2^{-6} \cdot \dfrac{m}{8} & e = 0 \text{ (subnormal)} \\[4pt]
(-1)^s \cdot 2^{\,e-7} \cdot \left(1 + \dfrac{m}{8}\right) & 1 \le e \le 15,\ (e,m) \ne (15,7) \\[4pt]
\text{NaN} & (e,m) = (15,7)
\end{cases}
$$

There is **no infinity**: the two reserved NaN codes aside, every other
`(e, m)` combination — including `e = 15` — decodes to a finite value. The
largest finite magnitude is $2^{8}\cdot 1.75 = 448$. **Encoding** a real
number picks the *nearest* representable finite E4M3 value (round-to-nearest;
this automatically saturates any magnitude above 448 to $\pm 448$, since 448
is then simply the closest grid point).

### bfloat16 rounding

`bfloat16` keeps float32's sign+exponent and truncates the mantissa to 7
bits, with round-to-nearest-even at the 16th bit. Emulate it on a float32
value by manipulating its raw bits:

```
bits32 = float32_value.view(uint32)
bias   = ((bits32 >> 16) & 1) + 0x7FFF
bits32 = (bits32 + bias) & 0xFFFF0000
bf16_emulated = bits32.view(float32)   # 16 low bits are now zero
```

## Task

Implement `cast_and_matmul_fp8e4m3`:

```python
def cast_and_matmul_fp8e4m3(W: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

* `W` — float32 weight matrix of shape `(k, n)`.
* `X` — float32 activation matrix of shape `(m, k)`, already on the bfloat16
  grid (i.e. `to_bf16(X) == X`).

Steps:
1. Encode `W` to E4M3 storage codes (`uint8` array, same shape as `W`, values
   `0..255`) using nearest-value rounding as defined above.
2. Decode those codes back to the E4M3 grid value, then round that value to
   `bfloat16` precision (the "upcast for compute" step) — call the result
   `W_bf16`.
3. Round `X` to `bfloat16` precision as well — call it `X_bf16` (a no-op here
   since `X` is already on the grid, but do it for correctness).
4. Compute `Y = X_bf16 @ W_bf16` with float32 accumulation.

Return `(Y, codes)` where `codes` is the `uint8` E4M3 storage array from step 1.

## Example

```python
import numpy as np
W = np.array([[1.0, -2.5], [0.3, 448.0]], dtype=np.float32)
X = np.array([[1.0, 0.0]], dtype=np.float32)
Y, codes = cast_and_matmul_fp8e4m3(W, X)
# codes.dtype == np.uint8, codes.shape == W.shape
# Y[0, 0] is ~1.0 (the E4M3/bf16-rounded reconstruction of W[0,0])
```

## What the gate checks

* **rel_err** — the global relative L2 error between your `Y` and the output
  of a NumPy oracle implementing the exact E4M3 encode/decode + bfloat16
  rounding + matmul pipeline above.
* **size_ratio** — `W.nbytes` (float32, 4 bytes/element) divided by the
  actual byte size of the `codes` array you return (should be `uint8`, 1
  byte/element, for a `4.0×` storage reduction). Returning codes in a wider
  dtype (e.g. `int32`) fails this gate even if the values are numerically
  correct.
