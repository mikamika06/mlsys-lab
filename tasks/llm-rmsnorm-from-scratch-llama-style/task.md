## Context

RMSNorm is a normalization technique that scales each element of an input vector by the root‑mean‑square (RMS) of all elements. For a vector $x \in \mathbb{R}^n$ and a learnable weight vector $w \in \mathbb{R}^n$, Llama‑style RMSNorm is defined as

$$
y_i = w_i \;\frac{x_i}{\sqrt{\tfrac{1}{n}\sum_{j=1}^{n}x_j^{2} + \varepsilon}},
$$

where $\varepsilon$ is a small constant added for numerical stability (typically $10^{-6}$).  No mean subtraction or bias term is used.

## Task

Implement the function `rmsnorm(x, weight, eps=1e-6)` that takes two NumPy arrays of identical shape and returns the RMS‑normalized result. The input may be any numeric dtype; the output should have the same dtype as the input.  Use only NumPy operations—no explicit Python loops.

## Example

```python
import numpy as np
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, 1.0, 1.5])
y = rmsnorm(x, w)
print(y)   # [0.40824829 0.81649658 1.22474487]
```

## What the gate checks

The grader computes a reference implementation using NumPy and compares your output with it via the scorer `max_abs_err`.  The maximum absolute difference must be at most $10^{-6}$.  
Additionally, the Jacobian of the function is numerically estimated by central finite differences; its maximum absolute error compared to the analytic Jacobian (derived from the formula above) must not exceed $5\times10^{-5}$.
