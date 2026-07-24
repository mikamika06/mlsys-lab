## Context

In deep neural networks, *depth pruning* removes entire blocks of layers from a model.  
If a network has $L$ layers with parameter counts $\{p_1,p_2,\dots ,p_L\}$, dropping the first $k$ layers eliminates

$$P_{\text{removed}} = \sum_{i=1}^{k} p_i.$$

The *fraction of parameters removed* is then

$$f_{\text{rem}} = \frac{P_{\text{removed}}}{P_{\text{total}}},\qquad
P_{\text{total}}=\sum_{i=1}^{L}p_i,$$

and the *remaining fraction* is simply $1-f_{\text{rem}}$.

These two numbers are useful for estimating how much of a model’s capacity and computational cost (FLOPs) will be lost when pruning layers from the front of the network.

## Task

Implement `removed_and_remaining`:

```python
def removed_and_remaining(param_counts: np.ndarray, k: int) -> tuple[float, float]:
    ...
```

- `param_counts` is a 1‑D NumPy array of non‑negative integers giving the number of parameters in each layer, ordered from first to last.
- `k` is the number of layers to drop from the **beginning** of the network (0 ≤ k ≤ len(param_counts)).
- The function must return a tuple `(removed_fraction, remaining_ratio)` where:
  - `removed_fraction = P_removed / P_total`
  - `remaining_ratio   = (P_total - P_removed) / P_total`

The implementation must use only NumPy operations and no explicit Python loops.

## Example

```python
import numpy as np
pc = np.array([100, 200, 300, 400])
k = 2
removed_fraction, remaining_ratio = removed_and_remaining(pc, k)
print(removed_fraction)   # 0.3
print(remaining_ratio)    # 0.7
```

## What the gate checks

The grader computes the reference values with NumPy and compares them to your output using the mean‑squared error (MSE).  
Your solution must achieve an MSE ≤ $10^{-12}$ on all test cases; otherwise it fails.
