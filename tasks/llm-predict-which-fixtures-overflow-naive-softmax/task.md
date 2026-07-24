## Context

Softmax is defined as  
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}\,. $$
In floating‑point arithmetic the exponential can overflow when its argument is too large. For IEEE 754 single precision ($\texttt{float32}$) the largest finite value is $\approx 3.4\times10^{38}$, so  
$$e^x \text{ overflows if } x > \log(3.4\times10^{38}) \approx 88.722839\,.$$
The same idea applies to other dtypes.

## Task

Implement `predict_overflow(logits_list)` that receives a list of one‑dimensional NumPy arrays containing logits and returns a list of booleans indicating whether the naive computation of softmax would overflow in **float32** for each array.

The function signature must be:

```python
def predict_overflow(logits_list: list[np.ndarray]) -> list[bool]:
    ...
```

## Example

```python
import numpy as np
logits = [
    np.array([0, 1, 2]),          # no overflow
    np.array([90, -5, 10]),       # overflow (90 > 88.7228)
]
print(predict_overflow(logits))
# [False, True]
```

## What the gate checks

The grader compares your output with a reference computed by NumPy. The metric `exact_match` must be `1.0`. A wrong implementation that uses the wrong dtype or comparison sign will fail.
