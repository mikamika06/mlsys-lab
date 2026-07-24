## Context

The soft‑max function maps a vector of logits $z \in \mathbb{R}^k$ to a probability distribution

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{k}\exp(z_j)}.$$

Its logarithm, the *log‑softmax*, is often used in cross‑entropy loss because it avoids an intermediate division:

$$\operatorname{log\_softmax}(z)_i
   = z_i - \log\!\Bigl(\sum_{j=1}^{k}\exp(z_j)\Bigr).$$

When the logits contain large positive or negative values, computing $\exp(z)$ directly can overflow or underflow. A standard trick is to subtract the maximum logit before exponentiation:

$$
\operatorname{log\_softmax}(z)_i
= z_i - \bigl(\max_j z_j + \log\!\sum_{j}\exp(z_j-\max_j z_j)\bigr).
$$

This form is numerically stable for any real input.

## Task

Implement the function `log_softmax` that takes a 2‑D NumPy array of shape $(n, k)$ and returns an array of the same shape containing the log‑softmax values computed in a numerically stable way. The result must be of type `float64`.

```python
def log_softmax(x: np.ndarray) -> np.ndarray:
    ...
```

## Example

```python
import numpy as np
x = np.array([[0, 1, 2],
              [1000, 1000, 1000],
              [-1000, -999, -998]])
y = log_softmax(x)
print(y)
# [[-2.40760596 -1.40760596 -0.40760596]
#  [0.         0.         0.]
#  [-3.00000000 -2.00000000 -1.00000000]]
```

## What the gate checks

The grader computes a reference log‑softmax using NumPy’s stable formulation and compares it to your output with the metric `max_abs_err`. The candidate passes only if

$$\mathrm{max\_abs\_err} \le 10^{-9}.$$

Additionally, the function must return a `float64` array of the same shape as the input. A naive implementation that first computes soft‑max and then takes the logarithm will overflow for large logits and fail this gate.
