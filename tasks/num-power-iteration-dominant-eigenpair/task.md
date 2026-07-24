## Context

The power iteration algorithm finds the dominant eigenpair of a matrix $A \in \mathbb{R}^{n \times n}$ (the eigenvalue $\lambda$ with largest absolute value and its corresponding eigenvector $v$). Starting from a random initial vector $b_0$, the iteration computes:

$$b_{k+1} = \frac{A b_k}{\|A b_k\|_2}.$$

After $k$ iterations, $b_k$ converges to the eigenvector $v$ associated with the dominant eigenvalue $\lambda$. The Rayleigh quotient gives an estimate of the eigenvalue:

$$\lambda \approx \frac{b_k^T A b_k}{b_k^T b_k}.$$

For a real symmetric matrix, the power iteration converges linearly with rate $|\lambda_2/\lambda_1|$, where $\lambda_2$ is the second-largest eigenvalue in absolute value.

## Task

Implement `power_iteration(A, num_iter)`:

```python
def power_iteration(A: np.ndarray, num_iter: int) -> tuple[float, np.ndarray]:
    """Return dominant eigenpair via power iteration.

    Args:
        A: Square real matrix (n×n)
        num_iter: Number of iterations to perform

    Returns:
        (eigenvalue, eigenvector) where eigenvector is normalized (unit 2-norm)
    """
```

Use a fixed initial vector $b_0 = \mathbf{1}$ (all ones) normalized to unit length. Iterate exactly `num_iter` times as described above. Return the Rayleigh quotient as the eigenvalue estimate and the final normalized $b_k$ as the eigenvector.

## Example

```python
import numpy as np
A = np.array([[2, 1],
              [1, 2]])
eigenvalue, eigenvector = power_iteration(A, 100)
# eigenvalue ≈ 3.0, eigenvector ≈ [0.7071, 0.7071]
```

## What the gate checks

The gate computes the true dominant eigenvalue $\lambda^*$ via `np.linalg.eigvals` and takes the one with largest absolute value. It then calculates the relative error:

$$\text{rel\_err} = \frac{|\lambda_{\text{computed}} - \lambda^*|}{|\lambda^*|}.$$

The solution passes if $\text{rel\_err} < 10^{-6}$. A naive implementation that doesn't normalize correctly or uses a poor stopping criterion will likely fail.
