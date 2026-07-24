## Context

The IEEE‑754 standard defines several binary floating‑point formats.  
The most common are **float32** (single precision), **float16** (half precision) and **bfloat16** (brain float).  

A *bit pattern* is the raw 32 or 16 bits that represent a number in memory.  
For example, the float32 value `1.0` has the bit pattern

$$
\texttt{0x3f800000}
$$

and its half‑precision counterpart has the bit pattern

$$
\texttt{0x3c00}.
$$

Converting a 32‑bit number to 16 bits involves rounding according to the IEEE rules.  
The *bfloat16* format keeps the sign and exponent of float32 but truncates the lower 16 fraction bits, so its bit pattern is simply the upper half of the float32 representation.

Python’s NumPy library provides convenient vectorised conversions:

```python
np.float32_array.astype(np.float16).view(np.uint16)   # fp16 bits
np.float32_array.view(np.uint32) >> 16                # bf16 bits (high 16 bits)
```

These operations are fast and produce the exact bit patterns that a C implementation would generate.

## Task

Implement the function `encode_fp32_to_fp16_and_bf16`:

```python
def encode_fp32_to_fp16_and_bf16(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `arr` is a one‑dimensional NumPy array of dtype `float32`.  
* The function must return a tuple `(fp16_bits, bf16_bits)` where each element is a 1‑D `np.uint16` array containing the IEEE bit patterns for the corresponding format.

The implementation must be fully vectorised (no Python loops) and should work on any platform that supports NumPy.

## Example

```python
import numpy as np

arr = np.array([0.0, 1.0, -2.5, 65504.0], dtype=np.float32)
fp16_bits, bf16_bits = encode_fp32_to_fp16_and_bf16(arr)

print(fp16_bits)   # array([    0, 15360,  49152, 65535], dtype=uint16)
print(bf16_bits)   # array([    0, 15360,  49152, 65535], dtype=uint16)
```

The printed values are the exact bit patterns that NumPy would produce for these numbers.

## What the gate checks

Two metrics are evaluated:

1. **fp16** – The byte‑exact fraction between the candidate’s `fp16_bits` and the reference bits obtained from `arr.astype(np.float16).view(np.uint16)`.  
   It must be exactly `1.0`.

2. **bf16** – The byte‑exact fraction between the candidate’s `bf16_bits` and the reference bits obtained from `(arr.view(np.uint32) >> 16).astype(np.uint16)`.  
   It must also be exactly `1.0`.

Both metrics are computed with `arena.scorers.byte_exact_fraction`, which returns a value in `[0, 1]`.  
A solution that fails to produce the exact bit patterns will receive a score of `0` for the corresponding metric and fail the gate.
