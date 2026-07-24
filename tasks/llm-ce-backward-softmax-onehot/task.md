## Context

For a classifier with logits $z \in \mathbb{R}^{C}$, the softmax function converts logits into probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} .
$$

The cross-entropy loss for a target class $y$ is

$$
L(z, y) = -\log(p_y).
$$

The gradient of cross-entropy with respect to the logits has a simple form. If $e_y$ is a one-hot vector with a $1$ at the target class and $0$ elsewhere, then

$$
\frac{\partial L}{\partial z} = p - e_y .
$$

This form is preferred in implementations because it avoids explicitly building intermediate Jacobian matrices.

## Task

Implement `ce_backward(logits, labels)`:

```python
def ce_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ...
```

The input `logits` is a 2-D NumPy array of shape $(N, C)$ containing model outputs before softmax. The input `labels` is a 1-D integer NumPy array of length $N$ containing the target class index for each row.

Return a float64 array of shape $(N, C)$ containing the gradient of the mean cross-entropy loss with respect to `logits`.

The implementation should use the stable softmax calculation by subtracting the row maximum before exponentiation.

## Example

```python
import numpy as np

logits = np.array([[2.0, 1.0, 0.0]])
labels = np.array([0])

grad = ce_backward(logits, labels)
# approximately:
# [[-0.3348, 0.2447, 0.0900]]
```

## What the gate checks

The gate computes a numerical gradient oracle using central finite differences on the cross-entropy loss. The returned gradient is compared against this oracle using relative L2 error.

A solution passes when

$$
\mathrm{rel\_err} =
\frac{\lVert g_{\mathrm{candidate}} - g_{\mathrm{oracle}}\rVert}
{\lVert g_{\mathrm{oracle}}\rVert + 10^{-12}}
< 10^{-4}.
$$
