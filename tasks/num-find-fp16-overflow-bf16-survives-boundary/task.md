## Context

The IEEE‑754 half precision format (FP16) has a maximum finite value of $65504$.  
When a number larger than this is cast to FP16 it becomes $\infty$.

Bfloat16 ($\mathrm{bf16}$) uses the same exponent range as FP32 but only 7 fraction bits.  
Its largest finite value is also $65504$, yet its rounding behaviour differs from FP16, so there can be a small interval of numbers that overflow in FP16 but remain finite in BF16.

Finding this boundary is useful when converting high‑precision tensors to low‑precision formats: it tells you the exact point at which precision loss turns into an infinite value for one format while still being representable by another.

## Task

Implement `find_fp16_overflow_boundary()`:

```python
def find_fp16_overflow_boundary() -> float:
    ...
```

It should return the smallest positive real number $x$ such that

$$\text{np.float16}(x) = \infty \quad\text{and}\quad \text{np.bfloat16}(x) \neq \infty.$$

If `np.bfloat16` is not available in the runtime, use `np.float32` as a surrogate (its range is far larger than FP16’s).  
The function must use only NumPy operations and no explicit Python loops.  
Return the value as a native Python `float`.

## Example

```python
import numpy as np
from your_module import find_fp16_overflow_boundary

boundary = find_fp16_overflow_boundary()
print(boundary)          # e.g. 65504.00048828125
print(np.float16(boundary))   # inf
print(np.bfloat16(boundary))  # 65504.0 (finite)
```

## What the gate checks

The grader computes a reference boundary using NumPy’s own casting rules and compares your result with an exact match (`==`).  
If the returned value differs, the solution fails.
