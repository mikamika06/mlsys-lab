## Context

The softmax function is ubiquitous in machine learning for converting a vector of logits $z \in \mathbb{R}^d$ into a probability distribution:

$$\operatorname{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.$$

When the entries of $z$ are large, $\exp(z_i)$ can overflow to $+\infty$, producing non‑finite values in the numerator or denominator. A *naïve* implementation that directly applies `np.exp` and divides by the sum will therefore return a matrix containing `inf` or `nan`.  
A *stable* variant subtracts the maximum logit before exponentiation:

$$\operatorname{softmax}_{\text{stable}}(z)_i = \frac{\exp(z_i - m)}{\sum_{j=1}^{d}\exp(z_j - m)}, \qquad m=\max_k z_k,$$

which keeps all intermediate values finite.

In this task you will write a helper that counts how many non‑finite entries appear when the naïve softmax is applied to a batch of logits. This count can be used as a diagnostic for numerical stability issues in larger pipelines.

## Task

Implement `count_nonfinite_in_naive_softmax`:

```python
def count_nonfinite_in_naive_softmax(x: np.ndarray) -> int:
    ...
```

* `x` is a 2‑D NumPy array of shape $(n, d)$ containing logits.
* The function must compute the naïve softmax (direct exponentiation and division) **using only NumPy**.
* It should return an integer equal to the number of elements in the resulting matrix that are not finite (`np.isfinite` returns `False`).
* All intermediate computations should use `float64`.

## Example

```python
import numpy as np
from solution_ref import count_nonfinite_in_naive_softmax

# Large logits that cause overflow
X = np.array([[1000, 1000, 1000],
              [2000, 2000, 2000]], dtype=np.float64)

cnt = count_nonfinite_in_naive_softmax(X)
print(cnt)   # → 6
```

The naïve softmax produces `inf` for every entry in the first row and also for the second row, so all six entries are counted.

## What the gate checks

A single exact‑match gate compares your function’s output on a fixed overflow test case to an oracle computed with NumPy. The expected integer count must match exactly; otherwise the submission fails.
