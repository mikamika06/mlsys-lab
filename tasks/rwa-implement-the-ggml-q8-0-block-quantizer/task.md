## Context

Quantization is a common technique in deep learning to reduce memory usage and accelerate inference.  
The **Q8_0** format used by the *ggml* library stores data in blocks of 32 elements.  
For each block it keeps:

1. A **scale** stored as a half‑precision float (`float16`).  
   The scale is chosen so that the largest absolute value in the block fits into the signed 8‑bit range:
   $$
   s = \frac{\max_{i} |x_i|}{127}\,,
   $$
   with the convention that if all values are zero we set $s=1.0$ to avoid division by zero.

2. An array of **int8** quantized coefficients
   $$
   q_i = \operatorname{clip}\!\bigl(\,\operatorname{round}(x_i / s),\,-127,\,127\bigr)\,.
   $$

The dequantised value is obtained by multiplying the integer back with the scale:
$$
\hat{x}_i = q_i \times s\,.
$$

This simple scheme preserves a good approximation of the original data while using only 4 bytes per element (1 byte for the coefficient and 2 bytes for the scale, plus one padding byte in many implementations).

## Task

Implement the function `q8_0_quantize` that takes a **single** block of 32 floating‑point numbers (`np.ndarray` of shape `(32,)`) and returns a tuple:

```python
def q8_0_quantize(block: np.ndarray) -> Tuple[np.ndarray, np.float16]:
    ...
```

* `block`: a one‑dimensional NumPy array of length 32 containing arbitrary real values (dtype may be any numeric type).
* Return value:
  * An **int8** NumPy array of shape `(32,)` holding the quantised coefficients.
  * A **float16** scalar representing the scale used for this block.

The implementation must follow exactly the formulas above, including clipping to `[-127, 127]`.  
It should work correctly for any numeric dtype input and produce the expected dtypes in the output.

## Example

```python
import numpy as np

block = np.array([0.0, 1.5, -2.3, 4.7, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

q, s = q8_0_quantize(block)
print(q.dtype, s.dtype)          # int8 float16
print(s)                         # 0.037007... (approx 4.7/127)
print(np.round(block / s).astype(int))   # should match q
```

## What the gate checks

The grader evaluates your function on a set of random blocks and computes the **mean squared error** between each original block and its dequantised reconstruction:

$$
\text{MSE} = \frac{1}{32}\sum_{i=0}^{31}(x_i - \hat{x}_i)^2\,.
$$

The solution must achieve a global MSE not exceeding $10^{-6}$ across all test blocks.  
Additionally, the grader verifies that:

* The returned coefficient array has dtype `np.int8` and shape `(32,)`.
* The scale is a scalar of dtype `np.float16`.

If any of these conditions fail, the gate will reject the submission.
