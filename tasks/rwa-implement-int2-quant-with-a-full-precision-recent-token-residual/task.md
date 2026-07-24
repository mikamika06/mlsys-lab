## Context

Quantization reduces the precision of numerical data to save memory and accelerate computation.  
A common scheme is **int‑2 quantization**, which maps real numbers onto four discrete levels.  
Given a segment of values $x_1,\dots,x_m$, we compute its minimum $\min$ and maximum $\max$ and map each value linearly to an integer in $\{0,1,2,3\}$:

$$
c_i = \operatorname{clip}\!\left(\bigl\lfloor (x_i-\min)\tfrac{3}{\max-\min} + 0.5\bigr\rceil,\;0,\;3\right).
$$

The de‑quantized value is then recovered as  

$$
\hat{x}_i = \min + \tfrac{c_i}{3}\,(\max-\min).
$$

In many language models a **residual window** of the most recent tokens is kept in full precision (e.g. fp16) to preserve accuracy for the latest context while still compressing older tokens.

## Task

Implement the function

```python
def int2_quant_with_residual(x: np.ndarray, R: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `x` – a one‑dimensional array of real numbers (any dtype convertible to float32).  
* `R` – number of most recent tokens that must be kept in fp16 precision.

The function should return a tuple `(codes, residuals)`:

1. **codes** – a 1‑D `uint8` array of length `max(0, len(x)-R)`.  
   Each element is an integer in `{0,1,2,3}` obtained by the int‑2 mapping described above applied to the first `len(x)-R` elements of `x`.

2. **residuals** – a 1‑D `float16` array containing the last `R` elements of `x`.  
   If `R >= len(x)`, `codes` must be an empty array and `residuals` contains all of `x` cast to `float16`.

The implementation must use only NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
from your_module import int2_quant_with_residual

x = np.array([0.1, 0.5, -0.3, 1.2, 0.8], dtype=np.float32)
R = 2
codes, residuals = int2_quant_with_residual(x, R)

print(codes)      # e.g. array([0, 3, 1], dtype=uint8)
print(residuals)  # array([-0.300003,  1.199997], dtype=float16)
```

Re‑quantizing `codes` with the same min/max and adding back `residuals` (cast to float64) should reconstruct `x` within a small numerical error.

## What the gate checks

The grader verifies two properties:

* **codes_match** – the returned `codes` array must be identical to that produced by an oracle implementation.  
  The metric is `1.0` if equal, otherwise `0.0`.

* **residual_len_ok** – the length of the `residuals` array must exactly equal the supplied `R`.  
  The metric is `1.0` if correct, otherwise `0.0`.

Both metrics are required to be `1.0` for a passing solution.
