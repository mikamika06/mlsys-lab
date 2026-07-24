## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^d$ to a probability distribution
$$
\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}\,.
$$
Its logarithm, the log‑softmax,
$$
\log \operatorname{softmax}(z)_i = z_i - \log\!\Bigl(\sum_{j=1}^{d} e^{z_j}\Bigr)\,,
$$
is frequently used in machine learning because it avoids an explicit division and is numerically more stable when combined with cross‑entropy loss.

A direct implementation of the log‑softmax can suffer from overflow or underflow if any component $z_i$ has a large magnitude. The standard trick is to subtract the maximum entry before exponentiating:
$$
\log \operatorname{softmax}(z)_i = z_i - m - \log\!\Bigl(\sum_{j=1}^{d} e^{\,z_j-m}\Bigr), \qquad
m=\max_{k} z_k .
$$
This guarantees that all exponentials are bounded by $e^0=1$ and the sum never exceeds $d$, preventing overflow.

## Task

Implement `stable_log_softmax`:

```python
def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    ...
```

The function receives a NumPy array of arbitrary shape containing real numbers. It must return an array of the same shape and dtype `float64`, where each slice along the specified `axis` has been transformed by the numerically stable log‑softmax formula above. No explicit Python loops are allowed; use only vectorized NumPy operations.

## Example

```python
import numpy as np
logits = np.array([[1, 2, 3], [4, 5, 6]])
D = stable_log_softmax(logits)
# [[-2.40760596 -1.40760596 -0.40760596]
#  [-2.40760596 -1.40760596 -0.40760596]]
```

## What the gate checks

The grader computes a reference implementation using NumPy’s own operations and compares your output with it via the metric `max_abs_err`. The maximum absolute difference must not exceed $10^{-10}$ on a set of random test cases that include very large positive and negative logits. A correct, fully vectorised solution will satisfy this tolerance; any overflow or incorrect broadcasting will fail.
