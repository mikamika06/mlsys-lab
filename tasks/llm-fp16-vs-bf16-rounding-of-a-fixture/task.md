## Context

In deep learning we often store tensors in reduced precision to save memory and bandwidth. The two most common formats are IEEE‑754 binary16 (FP16) and the truncated exponent‑mantissa format known as BF16 or bfloat16. Both have a 1‑bit sign, an 8‑bit exponent field, but differ in how many fraction bits they keep: FP16 keeps 10 fraction bits while BF16 keeps only 7. Consequently BF16 has a larger dynamic range but lower precision.

Rounding from a full‑precision float32 to one of these formats is performed by truncating the low‑order bits of the mantissa. In NumPy we can obtain an FP16 representation simply with `astype(np.float16)`. BF16 is not natively supported, so we must emulate it by masking off the lower 16 bits of each 32‑bit word and interpreting the remaining upper 16 bits as a new float32 value.

## Task

Implement a function that takes a one‑dimensional NumPy array of dtype `float32` and returns two arrays:

```python
def compare_rounding(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (fp16_arr, bf16_arr) where each element is the value obtained by
    rounding x to FP16 and BF16 respectively.  The returned arrays must be of
    dtype float32 so that they can be compared with the original input.
    """
```

The function should:

1. Convert `x` to FP16, then cast back to float32 for comparison.
2. Emulate BF16 by truncating the lower 16 bits of each 32‑bit word and
   interpreting the remaining upper 16 bits as a new float32 value.

## Example

```python
import numpy as np
from compare_rounding import compare_rounding

x = np.array([0.1, -2.5, 1234.5678], dtype=np.float32)
fp16_arr, bf16_arr = compare_rounding(x)

print(fp16_arr)   # array of float32 values rounded to FP16
print(bf16_arr)   # array of float32 values rounded to BF16
```

## What the gate checks

The grader computes a reference implementation using NumPy and bit‑twiddling.  
It then evaluates two metrics:

- `max_fp16_err`: the maximum absolute difference between the returned FP16
  array and the reference.
- `max_bf16_err`: the same for BF16.

Both metrics must be less than or equal to the thresholds specified in the
task metadata (1e‑5 for FP16, 1e‑4 for BF16). A correct implementation will
match the reference exactly, yielding zero error; any deviation will cause a
gate failure.
