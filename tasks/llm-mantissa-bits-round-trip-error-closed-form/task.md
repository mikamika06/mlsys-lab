## Context

Floating‑point numbers are represented in binary as  

$$x = \pm\,2^{e}\,\bigl(1.b_1b_2\ldots b_m\bigr) ,$$

where $m$ is the number of mantissa bits.  
When a real value is rounded to this format, the relative error is bounded by the *unit roundoff*

$$u = 2^{-(m+1)}.$$

If we convert a double‑precision value to a lower‑precision format and then back again (a **round‑trip**), the only source of error is the first conversion; the second step merely represents the already rounded value exactly.  
Thus the maximum relative round‑trip error for a given mantissa size $m$ is

$$\varepsilon_{\max}(m) = 2^{-(m+1)}.$$

## Task

Implement `round_trip_error` that, given an integer or array of integers representing mantissa bit widths, returns the corresponding maximum relative round‑trip errors as a NumPy array of type `float64`.

```python
def round_trip_error(mantissa_bits: int | Sequence[int]) -> np.ndarray:
    ...
```

The function must work for scalar inputs and for one‑dimensional sequences.  
No explicit Python loops are required, but the implementation may use broadcasting.

## Example

```python
import numpy as np
from llm_mantissa import round_trip_error  # your module name

print(round_trip_error(10))
# [4.8828125e-04]

print(round_trip_error([8, 12, 20]))
# [1.9531250e-03 3.0517578e-04 7.6293945e-07]
```

## What the gate checks

The grader computes a reference value using the closed‑form formula above and compares it to your output with the metric `rel_err` (global relative L2 error).  
Your solution must satisfy  

$$\mathrm{rel\_err} \le 10^{-9}.$$

A wrong implementation that, for example, forgets the `+1` in the exponent will produce a relative error of about $1$ and fail this gate.
