## Context

The eight‑bit *e4m3* format stores a floating point number with one sign bit, four exponent bits and three mantissa bits.  
For an 8‑bit word `b7 b6 b5 b4 | b3 b2 b1 b0` the decoded value is

$$
(-1)^{b_7}\times
\begin{cases}
0 & \text{if }b_{3..0}=0000\\[4pt]
\dfrac{b_{3..0}}{8}\times 2^{-6} & \text{if }b_{3..0}\neq 0000,\, b_{6..4}=000\\[6pt]
\left(1+\dfrac{b_{3..0}}{8}\right)\times 2^{\,b_{6..4}-7} & \text{otherwise}
\end{cases}
$$

The representable values are the set of all numbers that can be expressed by this formula.  
Values with absolute magnitude greater than $448$ do not fit in the format and would
*saturate* to $\pm\infty$ in a typical conversion routine.

A value is considered **exact** if it belongs exactly to the representable set.
If it does not belong but its magnitude lies within the normal range,
the nearest representable number is a *subnormal* value.  Values beyond $448$
are classified as **saturated**.

## Task

Implement `classify_e4m3`:

```python
def classify_e4m3(vals: list[float]) -> list[str]:
    ...
```

It receives a list of floating point numbers (of any shape).  
Return a one‑dimensional string array where each element is either `"EXACT"`,
`"SUBNORMAL"` or `"SATURATED"`, indicating the classification of the
corresponding input value.

The implementation must use only pure Python and Python.  It should not
hardcode the classification of individual numbers; instead, it should perform
the conversion to e4m3 and compare against the original value.

## Example

```python
from classify import classify_e4m3

vals = [0.0, 0.01, 1.0, 1000.0, float('nan')]
print(classify_e4m3(vals))
# ['EXACT' 'SUBNORMAL' 'EXACT' 'SATURATED' 'SATURATED']
```

## What the gate checks

The solution is graded by comparing its output to a reference implementation
that performs the same conversion.  The gate requires **exact match** of all
elements in the returned array for the provided test cases.
