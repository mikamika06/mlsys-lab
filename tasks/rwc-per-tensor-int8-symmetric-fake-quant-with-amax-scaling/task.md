## Context

In many deep‑learning frameworks weights are stored in a reduced precision format to save memory and accelerate inference.  
A common scheme is **symmetric per‑tensor INT8 fake quantization**: each tensor is scaled by its maximum absolute value (amax) so that the range $[-\text{amax},\,\text{amax}]$ maps onto the integer interval $[-127,\,127]$.  

The scaling factor is

$$
s = \frac{\text{amax}}{127},
$$

and a real weight $x$ is quantized to an integer code $q$ by

$$
q = \operatorname{clip}\!\bigl(\,\operatorname{round}\!\bigl(\tfrac{x}{s}\bigr),\,-127,\,127\bigr).
$$

The de‑quantized value used during a forward pass is then

$$
x_{\text{dq}} = q \times s .
$$

This process is called *fake quantization* because the integer codes are never written to disk; they exist only temporarily in memory.

## Task

Implement the function

```python
def per_tensor_int8_symmetric_fake_quant(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

It receives a NumPy array `x` of arbitrary shape containing real numbers.  
The function must return a tuple `(codes, dequantized)` where:

* **codes** – an integer array of the same shape as `x`, dtype `np.int8`.  
  Each element is the quantized code computed with the algorithm above.
* **dequantized** – a float64 array of the same shape as `x` containing the
  de‑quantized values `q * s`.

Special cases:

* If all elements of `x` are zero, set `s = 1.0` to avoid division by zero;  
  then every code is zero and the de‑quantized array is also all zeros.

The implementation must use only NumPy operations – no explicit Python loops.

## Example

```python
import numpy as np
from your_module import per_tensor_int8_symmetric_fake_quant

x = np.array([0.0, -1.5, 2.3, -4.7])
codes, dq = per_tensor_int8_symmetric_fake_quant(x)
print(codes)      # array([-127,   95, -127], dtype=int8)
print(dq)         # array([-1.5 ,  1.5 , -4.7 ])
```

## What the gate checks

Two metrics are evaluated:

* **exact_codes** – compares the integer codes returned by your function to
  a reference implementation computed on the fly. The comparison is exact
  (`np.array_equal`). The gate passes if all test cases match.

* **max_abs_err** – computes the maximum absolute difference between your
  de‑quantized array and the reference de‑quantized array. The gate requires
  this error to be at most `1e-5`. This tolerates tiny floating‑point rounding
  differences but still guarantees that the de‑quantization is correct.

The grader runs a handful of random and edge‑case tensors; your solution must
satisfy both metrics for all cases.
