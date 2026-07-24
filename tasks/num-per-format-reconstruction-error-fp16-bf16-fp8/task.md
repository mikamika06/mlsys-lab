## Context

Low-precision floating-point formats reduce memory usage by representing values with
fewer bits. A reconstruction error measures the difference between an original
activation value $x$ and a decoded value $\hat{x}$.

The maximum absolute reconstruction error is

$$
\mathrm{max\_abs\_err}(x,\hat{x}) = \max_i |x_i-\hat{x}_i|.
$$

Different formats trade precision and exponent range. IEEE fp16 stores values with
16 bits. Bfloat16 keeps the fp32 exponent width while reducing the mantissa. The
fp8 E4M3 format uses one sign bit, four exponent bits, and three mantissa bits.

## Task

Implement these functions:

```python
def fp16_roundtrip(x: np.ndarray) -> np.ndarray:
    ...

def bf16_roundtrip(x: np.ndarray) -> np.ndarray:
    ...

def fp8_e4m3_roundtrip(x: np.ndarray) -> np.ndarray:
    ...
```

Each function receives a NumPy array and returns a `float32` array with the same
shape containing the decoded values after conversion to the target format and back.

Requirements:

- `fp16_roundtrip` must perform a conversion through NumPy `float16`.
- `bf16_roundtrip` must emulate bfloat16 by truncating the lower 16 bits of each
  fp32 value.
- `fp8_e4m3_roundtrip` must implement fp8 E4M3 quantization with exponent bias
  $7$, three mantissa bits, and finite values only. Values outside the representable
  range must saturate to the nearest finite value.

## Example

```python
import numpy as np

x = np.array([1.0, 0.5, 3.14159], dtype=np.float32)

a = fp16_roundtrip(x)
b = bf16_roundtrip(x)
c = fp8_e4m3_roundtrip(x)
```

The three outputs contain different approximations of the same activation array.

## What the gate checks

The grader creates activation values and computes the format conversions using an
independent conversion algorithm. It compares the returned arrays using

$$
\max_i |x_i-\hat{x}_i|.
$$

The fp16, bf16, and fp8 reconstruction errors must each be exactly zero against
the reference conversion.
