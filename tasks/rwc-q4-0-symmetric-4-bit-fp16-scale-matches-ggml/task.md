## Context

Quantization compresses a tensor of real numbers into a smaller representation while preserving its essential structure.  
In the GGUF format used by **ggml**, the *Q4_0* scheme stores each block of $32$ weights as a single 16‑bit floating point scale $d$ and four‑bit integer codes $c_i \in \{0,\dots,15\}$.  
The encoding is symmetric: the original weight $w_i$ is approximated by

$$
\hat w_i = (c_i-8)\; d .
$$

To obtain the code we first compute a divisor

$$
d = \frac{\max_{j=1}^{32}\lvert w_j\rvert}{8},
$$

then round and shift:

$$
c_i = \operatorname{clip}\!\bigl(\,\bigl\lfloor\,\tfrac{w_i}{d} + 0.5\bigr\rfloor + 8,\; 0,\;15\bigr).
$$

The clipping ensures that the four‑bit representation stays within its bounds.

## Task

Implement two functions that perform this quantization and dequantization:

```python
def q4_0_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantizes a 1‑D array of real weights into Q4_0 format.
    Returns a tuple (codes, scales) where:
      * codes   : uint8 array of length len(weights), values in [0,15]
      * scales  : float16 array of shape (len(weights)//32,) containing the
                  per‑block scale d used during encoding.
    The input must have a length that is a multiple of 32.
    """
```

```python
def q4_0_dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantizes Q4_0 codes back to real weights using the provided per‑block
    scales. Returns a float64 array of the same length as `codes`.
    """
```

Both functions must be fully vectorised (no explicit Python loops over individual elements).  
The implementation should match exactly the algorithm described in the context section.

## Example

```python
import numpy as np

weights = np.array([0.0, 1.2, -3.4, 5.6] + [0]*28)   # length 32
codes, scales = q4_0_quantize(weights)
recovered = q4_0_dequantize(codes, scales)

print("Codes:", codes[:4])          # e.g. [8, 12, 3, 15]
print("Scale :", scales[0])         # e.g. 0.425
print("Recovered:", recovered[:4])   # close to original values
```

## What the gate checks

The grader computes a reference implementation of the GGUF Q4_0 algorithm using NumPy and compares:
1. **Exact match** – the `codes` array returned by your function must be identical to the reference codes for every test case.
2. **Reconstruction quality** – after dequantising with your own `q4_0_dequantize`, the relative L₂ error against the original weights must not exceed $5\times10^{-1}$.

If both conditions hold for all generated test cases, the gate reports success (`exact_match == 1.0`). Otherwise it fails.
