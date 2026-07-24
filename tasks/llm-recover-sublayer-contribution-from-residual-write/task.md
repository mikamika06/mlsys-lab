## Context

In a transformer block the residual stream is updated by adding the sublayer output to the current state.  
If $x_{\text{in}}$ denotes the input of the block and $f(x_{\text{in}})$ the sublayer contribution, then the block writes

$$x_{\text{out}} = x_{\text{in}} + f(x_{\text{in}}).$$

Given only $x_{\text{in}}$ and $x_{\text{out}}$, recovering the sublayer output is a simple vector subtraction:

$$f(x_{\text{in}}) = x_{\text{out}} - x_{\text{in}}.$$

The task is to implement this recovery in pure NumPy.

## Task

Implement `recover_sublayer_contribution(in_, out_)` that takes two 1‑D or 2‑D NumPy arrays of equal shape and returns the element‑wise difference `out_ - in_`. The result must be a NumPy array of type `float64`.

```python
def recover_sublayer_contribution(in_: np.ndarray, out_: np.ndarray) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np
in_  = np.array([0.5, -1.2, 3.4])
out_ = np.array([1.5, -0.8, 2.9])
sub = recover_sublayer_contribution(in_, out_)
print(sub)          # [ 1.   0.4 -0.5]
```

## What the gate checks

The grader computes a reference by subtracting the two arrays with NumPy and then evaluates the maximum absolute error:

$$\max_{i} |\, \text{candidate}[i] - \text{reference}[i] \,|.$$

The solution must achieve `max_abs_err <= 1e-7`. Any shape mismatch or incorrect arithmetic will cause a failure.
