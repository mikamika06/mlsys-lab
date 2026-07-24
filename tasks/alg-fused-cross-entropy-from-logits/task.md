## Context

The cross‑entropy loss for a single example with logits $z \in \mathbb{R}^C$ and an integer target class $t\in\{0,\dots,C-1\}$ is
$$
\ell(z,t) = -\,\log\!\bigl(\frac{\exp(z_t)}{\sum_{c=0}^{C-1}\exp(z_c)}\bigr)
          = -z_t + \log\!\Bigl(\sum_{c=0}^{C-1}\exp(z_c)\Bigr).
$$
Directly computing the denominator can overflow when some logits are large. The standard trick is to subtract the maximum logit $m=\max_c z_c$ before exponentiating:
$$
\log\!\Bigl(\sum_{c}\exp(z_c)\Bigr)
  = m + \log\!\Bigl(\sum_{c}\exp(z_c-m)\Bigr).
$$
Using this *log‑sum‑exp* trick yields a numerically stable implementation.

## Task

Implement `fused_cross_entropy(logits, targets)`:

```python
def fused_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    ...
```

`logits` is a 2‑D NumPy array of shape $(N,C)$ containing unnormalised scores for $N$ examples and $C$ classes.  
`targets` is a 1‑D integer array of length $N$ with the correct class index for each example.

The function must return the mean cross‑entropy loss over the batch as a scalar `float64`.  
It should be fully vectorised: no explicit Python loops are allowed, and it must use the log‑sum‑exp trick to avoid overflow.

## Example

```python
import numpy as np
logits = np.array([[0.2, 1.5, -0.3],
                   [2.0, 0.0, 0.1]])
targets = np.array([1, 0])
loss = fused_cross_entropy(logits, targets)
print(loss)   # ≈ 0.6931471805599453
```

## What the gate checks

The grader computes a reference loss with NumPy’s stable log‑sum‑exp routine and compares it to your implementation using the metric `max_abs_err`.  
Your solution must satisfy

$$\mathrm{max\_abs\_err} \le 10^{-9}.$$

Additionally, the return value must be a scalar of type `float64`; otherwise the gate fails.
