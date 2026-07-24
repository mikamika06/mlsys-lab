## Context

The IEEE‑754 single‑precision format encodes a real number in a 32‑bit unsigned integer. The most significant bit is the sign $s$, the next eight bits are the exponent $e$, and the remaining twenty‑one bits form the fraction (mantissa) $f$.

$$
x = (-1)^s \times 2^{\,e-127} \times \bigl(1.f\bigr)
$$

Special values arise when the exponent field is all zeros or all ones. The table below lists the five canonical categories that we will distinguish:

| Category | Exponent bits | Fraction bits | Meaning |
|----------|---------------|--------------|---------|
| Zero | $000\,0000_2$ | $0$ | $\pm 0$ |
| Subnormal | $000\,0000_2$ | $>0$ | $\pm$ subnormal |
| Normal | $1\,000\,001_2 \dots 111\,1111_2$ | any | $\pm$ normal |
| Infinity | $111\,1111_2$ | $0$ | $\pm\infty$ |
| NaN | $111\,1111_2$ | $>0$ | Not a Number |

The task is to classify an array of raw 32‑bit patterns into these five categories, preserving the sign. The output labels are encoded as integers:

$$
\text{label} = s \times 4 + c,
$$

where $c\in\{0,\dots,4\}$ indexes the category in the order Zero, Subnormal, Normal, Infinity, NaN.

## Task

Implement `classify_uint32_patterns`:

```python
def classify_uint32_patterns(arr: np.ndarray) -> np.ndarray:
    ...
```

The argument is a 1‑D NumPy array of dtype `np.uint32`. The function must return a new array of the same shape and dtype `np.int8`, containing the integer labels described above. **No floating‑point operations or comparisons are allowed**; only integer bitwise manipulation may be used.

## Example

```python
import numpy as np
arr = np.array([0x00000000, 0x80000000, 0x00400000,
                0x3F800000, 0x7F800000, 0x7FC00000], dtype=np.uint32)
labels = classify_uint32_patterns(arr)
print(labels)
# [ 0  4  5 10 12 14]
```

Explanation:  
- `0x00000000` → positive zero → label $0$  
- `0x80000000` → negative zero → label $4$  
- `0x00400000` → positive subnormal → label $5$  
- `0x3F800000` → positive normal (1.0) → label $10$  
- `0x7F800000` → positive infinity → label $12$  
- `0x7FC00000` → NaN → label $14$

## What the gate checks

The grader computes a reference classification using NumPy’s float32 conversion and standard predicates (`np.isnan`, `np.isinf`, etc.). It then compares your output to that reference with an exact match. Your implementation must produce identical labels for all test patterns; otherwise the gate fails.

Additionally, the solution is required to use only integer operations on the bit patterns. Any use of floating‑point arithmetic or comparisons will cause the grader to reject the submission even if the numerical result happens to be correct.
