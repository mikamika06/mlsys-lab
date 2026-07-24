## Context

The softmax function maps logits $z \in \mathbb{R}^k$ to probabilities:

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} .
$$

When logits are stored in low precision such as float16, the exponential can
overflow. For example, large positive values may produce $e^z = \infty$, which
can make the resulting probabilities invalid.

A numerically stable implementation subtracts the largest logit before computing
the exponentials:

$$
\mathrm{softmax}(z_i) =
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\quad
m = \max_j z_j .
$$

The subtraction changes all exponential values by the same positive scale factor,
so the index of the largest probability is unchanged.

## Task

Implement `stable_softmax_argmax(logits)`:

```python
def stable_softmax_argmax(logits: np.ndarray) -> np.ndarray:
    ...
```

The input is a 2-D NumPy array of shape $(n, k)$ with dtype `float16`. Return a
1-D integer NumPy array containing the index of the largest softmax probability
for each row.

Use a numerically stable softmax computation. Avoid directly computing
`exp(logits)` on the float16 values.

## Example

```python
import numpy as np

logits = np.array([
    [99, 100, 98],
    [-20, -10, -30],
], dtype=np.float16)

idx = stable_softmax_argmax(logits)
# array([1, 1])
```

## What the gate checks

The gate computes a reference answer using NumPy float64 stable softmax
arithmetic and compares the returned indices.

The metric $\mathrm{argmax\_agreement}$ is the fraction of rows whose returned
index matches the reference index:

$$
\mathrm{argmax\_agreement} = \frac{\text{matching rows}}{\text{total rows}} .
$$

The required value is

$$
\mathrm{argmax\_agreement} = 1.0 .
$$

A naive float16 implementation can overflow and return incorrect indices, while
a stable implementation matches the float64 reference.
