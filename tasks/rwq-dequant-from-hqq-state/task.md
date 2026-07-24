## Context

In many quantization schemes, the original floating‑point weight matrix \(W \in \mathbb{R}^{n\times m}\) is stored in a compact integer form.  
For HQQ (Hybrid Quantized Quantization) each column of \(W\) is represented by an unsigned integer code \(W_q\), a per‑column scale vector \(s\) and a zero‑point vector \(z\). The dequantization formula used by the library is

$$
W = (W_q - z)\; \odot\; s,
$$

where \(\odot\) denotes element‑wise multiplication.  
The goal of this task is to implement this operation correctly.

## Task

Implement `dequant_from_hqq_state`:

```python
def dequant_from_hqq_state(W_q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    ...
```

- `W_q` – a 2‑D NumPy array of unsigned integers (dtype can be any integer type).  
- `scale` – a 1‑D array of floats with length equal to the number of columns of `W_q`.  
- `zero` – a 1‑D array of integers with the same shape as `scale`.  

The function must return a NumPy array of dtype `float64` containing the dequantized weights.

## Example

```python
import numpy as np

W_q   = np.array([[0, 255], [128, 64]], dtype=np.uint8)
scale = np.array([0.5, 2.0])
zero  = np.array([10, 20])

W = dequant_from_hqq_state(W_q, scale, zero)
print(W)
# [[-5.  470.]
#  [59.  88.]]
```

## What the gate checks

The grader computes a reference result with NumPy and compares it to your output using the scorer `max_abs_err`.  
Your implementation must satisfy

$$
\max_{i,j} |\, \hat{W}_{ij} - W_{ij}\,| \le 10^{-6}.
$$

Any larger error will cause the gate to fail.
