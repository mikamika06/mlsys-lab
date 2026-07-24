## Context

Q4_K is a low-bit weight format used by production inference systems. A row is
processed in super-blocks of 256 values. Each super-block contains 8 sub-blocks
of 32 values.

Each sub-block stores a 6-bit scale and a 6-bit minimum. The scale and minimum
values are themselves quantized with two FP16 super-block values $d$ and
$d_{\min}$.

For a value with 4-bit quantized code $q$ and 6-bit minimum code $m$, the
dequantization rule is

$$
w = d \cdot s \cdot q - d_{\min} \cdot m ,
$$

where $s$ is the unpacked 6-bit scale code. The minimum code controls the
offset, allowing asymmetric quantization.

The 256 4-bit codes are packed two per byte. The eight scale codes and eight
minimum codes are packed together as sixteen 6-bit integers occupying 12 bytes.

## Task

Implement:

```python
def q4k_quantize_row(x: np.ndarray) -> tuple:
    ...
```

and

```python
def q4k_dequantize_row(codes: np.ndarray, scales_mins: np.ndarray,
                       d: np.ndarray, dmin: np.ndarray) -> np.ndarray:
    ...
```

`x` is a two-dimensional `float32` array. Its second dimension is always a
multiple of 256.

`q4k_quantize_row` must return:

- `codes`: `uint8` array of shape `(rows, columns // 2)` containing packed 4-bit
  values.
- `scales_mins`: `uint8` array of shape `(rows, columns // 256 * 12)` containing
  the packed 6-bit scale and minimum metadata.
- `d`: `float16` array containing one super-block scale per 256 values.
- `dmin`: `float16` array containing one super-block minimum scale per 256 values.

`q4k_dequantize_row` must reconstruct the original shape as a `float32` array.

The implementation should follow the Q4_K packing layout. Use NumPy operations for
the numerical work.

## Example

```python
import numpy as np

x = np.array([[0.0, 1.0, -2.0, 3.0] * 64], dtype=np.float32)

codes, scales_mins, d, dmin = q4k_quantize_row(x)
y = q4k_dequantize_row(codes, scales_mins, d, dmin)

# y has the same shape as x and contains the Q4_K reconstruction.
```

## What the gate checks

The gate builds its own NumPy Q4_K oracle. It checks that the packed 4-bit codes
and packed scale/minimum metadata match the oracle exactly on representative
weight rows.

It also dequantizes the student's packed representation and compares it with
the oracle reconstruction using maximum absolute error. The reconstruction error
must satisfy $\max_i |x_i-\hat{x}_i| \le 0.02$.
