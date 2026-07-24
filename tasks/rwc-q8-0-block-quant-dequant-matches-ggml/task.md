## Context

Block quantization is a common compression technique in machine‑learning inference libraries.  
For a vector $x \in \mathbb{R}^n$ we split it into blocks of size $B=32$.  
For each block we compute the maximum absolute value  

$$\text{absmax}_b = \max_{i \in \text{block } b}\lvert x_i\rvert,$$

and define a scale

$$d_b = \frac{\text{absmax}_b}{127}.$$

The integer code for each element is then

$$c_i = \operatorname{round}\!\left(\frac{x_i}{d_b}\right),\qquad c_i \in [-127,\,127].$$

In the GGML implementation $d_b$ is stored as a 16‑bit floating point number (float16) and the codes are signed 8‑bit integers.  
Dequantization simply multiplies each code by its block’s scale:

$$\hat{x}_i = c_i \, d_b.$$

The goal of this task is to implement these two steps exactly as GGML does.

## Task

Implement the following two functions in `starter.py` (and later in your solution):

```python
def q8_0_quantize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes a 1‑D array x into int8 codes and float16 scales.
    The length of x must be a multiple of 32.  Returns (codes, scales).
    """

def q8_0_dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantizes the block‑wise quantized data back to float32 values.
    Expects codes dtype=int8 and scales dtype=float16.
    """
```

* `x` is a 1‑D NumPy array of arbitrary numeric type; cast it to `float32`.  
* The output `codes` must have the same shape as `x` and dtype `np.int8`.  
* The output `scales` must be a 1‑D array of length `len(x)//32` with dtype `np.float16`.  
* Dequantization should return a float32 array of the same shape as `codes`.

## Example

```python
import numpy as np
x = np.array([0.0, 1.0, -2.5, 3.7] * 8, dtype=np.float32)   # length 32
codes, scales = q8_0_quantize(x)
print(codes.dtype, scales.dtype)          # int8, float16
x_hat = q8_0_dequantize(codes, scales)
np.testing.assert_allclose(x, x_hat, atol=1e-5)
```

## What the gate checks

The grader generates several random arrays whose lengths are multiples of 32.  
For each array it:

1. Calls your `q8_0_quantize` and compares the returned codes and scales **bit‑by‑bit** with a reference implementation.
2. Calls your `q8_0_dequantize` on those outputs and checks that the dequantized values match the reference exactly.

If any mismatch occurs, the gate fails.  The metric is named `exact_match`; it must equal `1.0` for success.
