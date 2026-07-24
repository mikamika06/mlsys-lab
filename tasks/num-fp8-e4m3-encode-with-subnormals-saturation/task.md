## Context

FP8 E4M3FN is a low precision floating point format with one sign bit, four exponent bits, and three fraction bits. The exponent bias is $7$.

For normal values, the represented number is

$$
(-1)^s \left(1+\frac{m}{2^3}\right)2^{e-7},
$$

where $e$ is the stored exponent field and $m$ is the fraction field. The exponent field $0$ is reserved for subnormal values, which use

$$
(-1)^s \frac{m}{2^3}2^{1-7}.
$$

The format has no infinity representation. Values outside the finite range must saturate. The largest finite magnitude is $448$.

Encoding a float32 value requires choosing the nearest representable FP8 value using round-to-nearest-even behavior, while preserving signed zero and handling NaNs.

## Task

Implement the functions:

```python
def encode_fp8_e4m3fn(x: np.ndarray) -> np.ndarray:
    ...

def decode_fp8_e4m3fn(codes: np.ndarray) -> np.ndarray:
    ...
```

`encode_fp8_e4m3fn` takes a NumPy array of float32 values and returns a NumPy array of dtype `uint8` containing E4M3FN bit patterns.

`decode_fp8_e4m3fn` takes a NumPy array of uint8 FP8 codes and returns a float32 array containing the decoded values.

The encoder must implement E4M3FN behavior including subnormals, saturation at magnitude $448$, signed values, and NaNs.

## Example

```python
import numpy as np

x = np.array([0.0, 1.0, 448.0, 500.0], dtype=np.float32)
codes = encode_fp8_e4m3fn(x)

# codes contain the FP8 byte representation
# decode_fp8_e4m3fn(codes) reconstructs the representable values
```

## What the gate checks

The encoder output is compared byte-for-byte against the `ml_dtypes.float8_e4m3fn` reference implementation. The `byte_exact_fraction` score must be $1.0$.

The decoder output is compared with the values obtained by converting the reference FP8 values back to float32. The maximum absolute error must satisfy

$$
\max_i |y_i-\hat{y}_i| \le 10^{-3}.
$$
