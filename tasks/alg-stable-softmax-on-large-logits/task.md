## Context
The standard formula for the softmax function for a vector $\mathbf{x}$ is given by:

$$ \text{softmax}(\mathbf{x})_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$

While mathematically correct, a naive implementation evaluates $\exp(x_i)$ directly. If some logits $x_i$ are very large (e.g., $x_i \approx 1000$), $\exp(x_i)$ will overflow the floating-point representation, resulting in `inf`, and the subsequent division will yield `nan` (Not a Number). 
To make it numerically stable, we exploit the shift-invariance property of softmax:

$$ \text{softmax}(\mathbf{x})_i = \text{softmax}(\mathbf{x} - c)_i $$

By choosing $c = \max(\mathbf{x})$, we can guarantee that all inputs to the exponentiation are $\le 0$, which prevents overflow.

## Task
Implement `stable_softmax(x)` where `x` is a 2D numpy array of shape `(N, D)` representing `N` independent distributions of `D` logits each. The function should compute the softmax along the second dimension (axis 1) using a numerically stable approach.

## Example
```python
import numpy as np

logits = np.array([[1000.0, 1001.0], [-1000.0, -999.0]])
print(stable_softmax(logits))
# Expected output:
# [[0.26894142 0.73105858]
#  [0.26894142 0.73105858]]
```

## What the gate checks
The gate computes the predicted probabilities on a dataset of extreme logit values. It checks the `mean_kl` metric: the mean KL divergence between your predicted probabilities and the numerically stable reference probabilities, which must be $\le 10^{-9}$.
