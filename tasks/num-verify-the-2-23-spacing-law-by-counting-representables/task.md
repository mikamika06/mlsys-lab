## Context

In IEEE‑754 single precision a real number is represented by a sign bit, an 8‑bit exponent and a 23‑bit fraction (mantissa). For any fixed exponent the spacing between adjacent representable numbers is constant; this distance is called one unit in the last place (ULP). In particular, for all $x$ with $1 \le x < 2$ the exponent field equals $127$, so there are exactly $2^{23}$ distinct mantissa patterns. Hence the interval $[1,2)$ contains precisely $2^{23}=8\,388\,608$ representable float32 values. The same holds for any interval of length $2^k$ that starts at a power of two; e.g. $[1024,2048)= [2^{10}, 2^{11})$ also has $2^{23}$ elements.

## Task

Implement `count_fp32_in_range(start: float, end: float) -> int`:

```python
def count_fp32_in_range(start: float, end: float) -> int:
    ...
```

The function receives two real numbers that are guaranteed to be representable as IEEE‑754 single precision and satisfy `start < end`. It must return the number of distinct float32 values $x$ such that $start \le x < end$. The result should be a Python integer.

## Example

```python
>>> count_fp32_in_range(1.0, 2.0)
8388608
>>> count_fp32_in_range(1024.0, 2048.0)
8388608
```

## What the gate checks

The grader calls your function twice with the ranges `[1,2)` and `[1024,2048)`. It compares the returned integers against a reference computed from NumPy’s bit‑level view of float32 numbers. The solution must match exactly; otherwise the `exact_match` metric is 0.0.
