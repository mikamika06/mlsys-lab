## Context

Fixed‑point arithmetic represents real numbers as integers scaled by a power of two.
For an $f$‑bit fractional part the scaling factor is $2^{\,f}$, and the quantized value
is obtained by rounding the product to the nearest integer.  
When the fractional part is exactly $\tfrac12$, the *banker’s rounding* rule,
also called **round‑half‑to‑even**, chooses the even integer:

$$Q(x) \;=\;\operatorname{round}_{\text{half‑to‑even}}\!\bigl(x\,2^{f}\bigr).$$

This mode avoids a systematic bias that would otherwise accumulate in long
computations.

## Task

Implement the function

```python
def quantize_fixed_point(arr: np.ndarray, frac_bits: int) -> np.ndarray:
    ...
```

It receives a NumPy array `arr` of arbitrary shape containing floating‑point
values and an integer `frac_bits` specifying how many fractional bits to keep.
The function must return an integer array of the same shape with dtype
`int64`, where each element is the result of applying the formula above.
Use only vectorized NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
A = np.array([0.125, 0.375])          # values that become .5 after scaling by 4
D = quantize_fixed_point(A, frac_bits=2)
print(D)   # [0 2]
```

The first element rounds $0.125\times4=0.5$ to the nearest even integer $0$,
the second rounds $1.5$ to $2$.

## What the gate checks

`check.py` generates a variety of random arrays and a special boundary case
where values are exactly at half‑integer boundaries after scaling.
For each test it computes the reference result with NumPy’s `np.round`
(which implements round‑half‑to‑even) and compares the student’s output
exactly.  The gate metric **exact_match** must equal `1.0` for all tests.
