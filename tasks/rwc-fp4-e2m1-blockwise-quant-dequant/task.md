## Context

Quantization compresses floating‑point tensors by mapping each element to a small set of discrete values.  
Bitsandbytes’ FP4 format uses a signed 4‑bit representation with an exponent of two bits and a mantissa of one bit (e2m1). The representable levels are the integers from –8 to +7, inclusive.  

For a tensor $X \in \mathbb{R}^{n}$ we perform *blockwise* quantization:  
1. Split $X$ into consecutive blocks of size $B$.  
2. For each block compute its absolute maximum $\alpha = \max_{i} |x_i|$.  
3. Scale the block by a factor $s = \frac{\alpha}{7}$ (the largest magnitude that can be represented).  
4. Quantize each element to the nearest integer in the set $\{-8,\dots,+7\}$:  
   $$q_i = \operatorname{clip}\!\bigl(\operatorname{round}(x_i / s), -8, 7\bigr)$$  
5. Dequantize by multiplying back with $s$:  
   $$\hat{x}_i = q_i \cdot s.$$

The algorithm is deterministic and can be implemented entirely with NumPy.

## Task

Implement the function:

```python
def fp4_quant_dequant(x: np.ndarray, block_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize `x` to FP4 e2m1 format in a blockwise manner and return both
    the integer codes (dtype=int8) and the dequantized float64 array.
    """
```

The function must:
- Accept any 1‑D or multi‑dimensional NumPy array of floats.  
- Treat the flattened array as a sequence of elements to be split into blocks of length `block_size`.  
- Return a tuple `(codes, dequant)` where `codes` is an int8 array of the same shape as `x`, and `dequant` is a float64 array of the same shape.  
- Use only NumPy; no Python loops over individual elements.

## Example

```python
import numpy as np
from fp4 import fp4_quant_dequant  # your implementation

x = np.array([0.0, 1.5, -2.3, 7.9, -8.0])
codes, deq = fp4_quant_dequant(x, block_size=2)
print(codes)   # e.g. [ 0,  2, -3,  7, -8]
print(deq)     # approx equal to original within the quantization error
```

## What the gate checks

The grader computes a reference implementation of the same algorithm and compares the student’s output **exactly**:  
- The integer codes must match element‑wise.  
- The dequantized values must match element‑wise (float64).  

If any mismatch occurs, the `exact_match` metric is set to 0.0; otherwise it is 1.0.
