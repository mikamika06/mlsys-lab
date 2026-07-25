## Context

Large language models often optimize cross-entropy loss over a vocabulary of logits. For
a vector of logits $z \in \mathbb{R}^{V}$, the cross-entropy loss for target token $y$
is

$$
L_{\mathrm{CE}} = -z_y + \log\left(\sum_{i=1}^{V} e^{z_i}\right).
$$

Very large logits can make training numerically unstable. A logit normalization
regularizer called z-loss adds an auxiliary penalty on the log partition value:

$$
L_{\mathrm{z}} = L_{\mathrm{CE}} + \lambda
\left(\log\left(\sum_{i=1}^{V} e^{z_i}\right)\right)^2 .
$$

The squared log-sum-exp term encourages the model to keep the normalization factor
near a controlled range while preserving the cross-entropy objective.

For numerical stability, compute log-sum-exp using the identity

$$
\log\left(\sum_i e^{z_i}\right)
=
m + \log\left(\sum_i e^{z_i-m}\right),
$$

where $m=\max_i z_i$.

## Task

Implement `z_loss(logits, targets, lambda_)`:

```python
def z_loss(logits: np.ndarray, targets: np.ndarray, lambda_: float) -> np.ndarray:
    ...
```

The input `logits` is a 2-D NumPy array of shape $(N, V)$ containing one row of
vocabulary logits per example. `targets` is an integer NumPy array of shape $(N,)$
containing the target vocabulary index for each row.

Return a 1-D NumPy array of shape $(N,)$ containing the combined loss for each
example:

$$
L_i =
-z_{i,y_i}
+
\log\left(\sum_j e^{z_{i,j}}\right)
+
\lambda
\left(
\log\left(\sum_j e^{z_{i,j}}\right)
\right)^2 .
$$

The implementation should use NumPy vectorized operations.

## Example

```python
import numpy as np

logits = np.array([[1.0, 2.0, 0.0],
                   [0.5, -1.0, 3.0]])
targets = np.array([1, 2])

out = z_loss(logits, targets, 0.01)
# array([0.028999..., 0.138...])
```

## What the gate checks

The gate computes a NumPy reference implementation of the numerically stable z-loss
formula and compares the submitted implementation against it.

The reported metric is `max_abs_err`. The maximum absolute difference between the
candidate output and the oracle output must satisfy

$$
\mathrm{max\_abs\_err} < 10^{-6}.
$$
