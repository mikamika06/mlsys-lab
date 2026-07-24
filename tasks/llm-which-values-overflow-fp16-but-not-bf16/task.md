## Context

Floating‑point formats differ in the range of values they can represent without overflow.  
The IEEE 754 binary16 format (fp16) has a maximum finite value of  

$$\text{max}_{\text{fp16}} = 65504.$$

Its bfloat16 counterpart (bf16) shares the same exponent width as float32, giving a much larger range:

$$\text{max}_{\text{bf16}} \approx 3.4\times10^{38}.$$

When a value exceeds its format’s maximum, casting to that type yields `inf`.  
The task is to identify which elements of an array would overflow when cast to fp16 but **not** when cast to bf16.

## Task

Implement the function

```python
def which_overflow_fp16_not_bf16(arr: np.ndarray) -> np.ndarray:
    ...
```

It receives a one‑dimensional NumPy array `arr` of arbitrary numeric type and returns a boolean array of the same shape.  
Each element is `True` if its absolute value exceeds `np.finfo(np.float16).max` **and** does not exceed the maximum representable value of bfloat16; otherwise it is `False`.

The implementation must use only NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
arr = np.array([0, 70000, -60000, 1e40, np.nan])
mask = which_overflow_fp16_not_bf16(arr)
print(mask)          # [False  True False False False]
```

The second element overflows fp16 but not bf16; the fourth would overflow both formats.

## What the gate checks

A single gate named `exact_match` compares your output to a reference computed by NumPy.  
Your function must return an array that matches the oracle exactly for all test cases.
