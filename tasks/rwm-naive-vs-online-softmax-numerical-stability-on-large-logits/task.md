## Context

Softmax converts a vector of logits into a probability distribution. For logits
$z \in \mathbb{R}^n$, the definition is

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} .
$$

A direct implementation is numerically unstable for large positive or negative
values because $e^{z_i}$ can overflow or underflow in floating point arithmetic.

Production implementations use the max-subtraction identity. Let

$$
m = \max_i z_i .
$$

Then

$$
\mathrm{softmax}(z)_i =
\frac{e^{z_i-m}}{\sum_{j=1}^{n} e^{z_j-m}} ,
$$

which produces the same mathematical result while keeping the exponent values
in a safer range.

For a matrix of logits $Z \in \mathbb{R}^{b \times n}$, softmax is applied
independently to each row.

## Task

Implement `stable_softmax(logits)`:

```python
def stable_softmax(logits: np.ndarray) -> np.ndarray:
    ...
```

The function takes a 2-D NumPy array of shape $(b, n)$ and returns a
`float64` NumPy array of the same shape. Each row must contain the softmax
distribution for the corresponding row of input logits.

Use the numerically stable max-subtraction algorithm. The implementation should
work for large-magnitude logits where directly computing `np.exp(logits)` would
overflow.

## Example

```python
import numpy as np

logits = np.array([[1000.0, 1001.0, 1002.0]])
probs = stable_softmax(logits)

# approximately:
# [[0.09003057, 0.24472847, 0.66524096]]
```

## What the gate checks

The gate computes a reference softmax using NumPy float64 arithmetic with the
max-subtraction formula. It tests the implementation on large-magnitude logits
where a naive exponentiation implementation overflows or loses precision.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y - y_{\mathrm{ref}} \rVert_2}
{\lVert y_{\mathrm{ref}} \rVert_2 + 10^{-12}}
$$

must satisfy the required threshold. The returned array must also remain
finite.
