## Context

Scaled dot‑product attention is defined by

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$

where $Q,K,V\in\mathbb R^{n\times d_k}$ are query, key and value matrices.  
In many production systems the keys and values are stored in a reduced‑precision format to save memory and bandwidth.  One popular choice is the **FP8 e4m3** format: an 8‑bit signed integer with 4 exponent bits and 3 mantissa bits.  The dynamic range of this format can be emulated by per‑tensor scaling:

$$
\text{scale}=\frac{\max(|X|)}{448}, \qquad
q = \operatorname{round}\!\left(\frac{X}{\text{scale}}\right), \qquad
X_{\text{deq}}= q\,\text{scale},
$$

where $X$ is either $K$ or $V$.  The constant $448$ comes from the maximum representable value of an e4m3 integer after rounding.

## Task

Implement a function that performs scaled dot‑product attention using per‑tensor FP8 e4m3 quantization on the key and value matrices:

```python
def quantized_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    ...
```

The function receives three 2‑D NumPy arrays of shape $(n,d_k)$ and returns an array of shape $(n,d_k)$ containing the attention output.  
All computations must use only NumPy; no explicit Python loops are allowed.

## Example

```python
import numpy as np
Q = np.array([[1,0],[0,1]], dtype=np.float32)
K = np.array([[1,2],[3,4]], dtype=np.float32)
V = np.array([[5,6],[7,8]], dtype=np.float32)

out = quantized_attention(Q,K,V)
print(out)
```

The printed matrix should be close to the exact attention result computed with full precision.

## What the gate checks

A single metric is evaluated:

* **max_abs_err** – the maximum absolute difference between the student's output and a NumPy oracle that implements the same algorithm.  The value must satisfy  
  $$\text{max\_abs\_err} \le 2\times10^{-2}.$$

The oracle quantizes $K$ and $V$ with per‑tensor scaling, performs softmax attention in full precision, and returns the result.  A naive implementation that does not use the correct scaling or uses loops will exceed this tolerance.
