## Context

In quantized neural networks each weight is stored with a fixed bit‑width $b$.  
When weights are grouped for per‑group scaling and biasing, the scale and bias
are typically represented as 16‑bit floating point numbers.  
If a group contains $g$ weights, the two FP16 parameters contribute an overhead of

$$\frac{32}{g} \text{ bits per weight}.$$

Thus the *effective* number of bits used to encode one weight is

$$b_{\mathrm{eff}} = b + \frac{32}{g}.$$

The task is to compute this quantity for arbitrary inputs.

## Task

Implement `effective_bits_per_weight(b, group_size)`:

```python
def effective_bits_per_weight(b: np.ndarray | int,
                              group_size: np.ndarray | int) -> np.ndarray:
    ...
```

`b` and `group_size` may be scalars or 1‑D arrays of the same shape.  
The function must return a NumPy array (or scalar) of type `float64`
containing $b_{\mathrm{eff}}$ for each pair.

## Example

```python
import numpy as np
# single weight, group size 4
print(effective_bits_per_weight(8, 4))
# 16.0

# batch of weights
b = np.array([8, 16])
g = np.array([1, 2])
print(effective_bits_per_weight(b, g))
# [40. 24.]
```

## What the gate checks

The grader evaluates your implementation against a NumPy oracle that computes  
$b_{\mathrm{eff}} = b + 32/g$ for several random test cases.
It reports the maximum absolute error `max_abs_err`.  
Your solution must achieve `max_abs_err <= 1e-9`.
