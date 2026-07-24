## Context

In many neural‑network inference engines activations are compressed to 8‑bit signed integers per token (row). The standard approach is symmetric quantization: for each row \(i\) we compute a scale factor
$$s_i = \frac{\max_j |A_{ij}|}{127}.$$
The integer code is then
$$q_{ij} = \operatorname{clip}\!\bigl(\,\mathrm{round}(A_{ij}/s_i),\, -128, 127\bigr).$$
Dequantisation recovers a floating‑point approximation
$$\hat A_{ij}=q_{ij}\, s_i.$$

The goal is to implement this per‑token quantiser efficiently in pure NumPy.

## Task

Implement `per_token_int8_quant(A)`:

```python
def per_token_int8_quant(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

It receives a 2‑D NumPy array of shape `(n, d)` and returns a tuple `(codes, scales)` where

* `codes` is an `(n, d)` array of dtype `np.int8`;
* `scales` is a 1‑D array of length `n` with dtype `float64`.

The implementation must use only NumPy vectorised operations; no explicit Python loops.

## Example

```python
import numpy as np
A = np.array([[0, -2, 3],
              [4, 0, -5]], dtype=np.float32)
codes, scales = per_token_int8_quant(A)
print(codes)
# [[  0 -128   127]
#  [  127    0 -128]]
print(scales)
# [0.01574803 0.03937008]   # approx
```

Dequantising with `codes.astype(np.float64) * scales[:, None]` reproduces the original values up to rounding error.

## What the gate checks

The grader computes a reference implementation using NumPy and compares your returned `codes` and `scales` exactly. The metric is `exact_match`; it must equal `1.0`. A correct implementation will therefore produce identical arrays for all test cases.
