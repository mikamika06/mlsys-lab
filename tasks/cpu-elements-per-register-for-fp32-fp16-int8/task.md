## Context

The number of SIMD lanes that can be stored in a register depends on the size (in bits) of the register and the element type being packed.  
If a register holds $B$ bits and each element occupies $b$ bits, then the integer number of elements that fit is  

$$L = \left\lfloor \frac{B}{b} \right\rfloor.$$  

For the common types we consider:

| Type | Bits per element |
|------|-------------------|
| `float32` | 32 |
| `float16` | 16 |
| `int8`    | 8 |

Thus a 128‑bit NEON register can hold $4$ `float32`, $8$ `float16`, and $16$ `int8` values.

## Task

Implement the function

```python
def lanes_per_register(reg_bits: int) -> dict:
    ...
```

It receives an integer `reg_bits` describing the width of a SIMD register (e.g. 128, 256 or 512).  
Return a dictionary that maps each data type name (`"float32"`, `"float16"`, `"int8"`) to the number of elements that fit in the given register.

The function should raise a `ValueError` if `reg_bits` is not positive.  The result must be an ordinary Python dictionary, keys exactly as shown, and values integer counts.

## Example

```python
>>> lanes_per_register(128)
{'float32': 4, 'float16': 8, 'int8': 16}
>>> lanes_per_register(256)
{'float32': 8, 'float16': 16, 'int8': 32}
```

## What the gate checks

The grading script compares your output with a reference implementation that computes the same division.  
A single gate named `exact_match` verifies that your dictionary is equal to the reference; any difference causes failure.
