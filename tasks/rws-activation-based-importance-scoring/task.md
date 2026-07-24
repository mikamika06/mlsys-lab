## Context

In many neural network architectures the relative importance of a neuron, head or channel can be estimated from its activations on a calibration set. A common heuristic is to take the mean absolute activation over all examples in the batch:

$$
I_j = \frac{1}{N}\sum_{i=1}^{N} |a_{ij}|
$$

where $a_{ij}$ denotes the activation of unit $j$ for example $i$, and $N$ is the batch size. The resulting vector $\mathbf I \in \mathbb R^C$ ranks units from most to least active.

## Task

Implement a function that computes this importance score.

```python
def score_importance(activations: np.ndarray) -> np.ndarray:
    """
    Compute mean absolute activation per unit.

    Parameters
    ----------
    activations : np.ndarray
        2‑D array of shape (batch, units). The values may be positive or negative.
    
    Returns
    -------
    importance : np.ndarray
        1‑D float64 array of length ``units`` containing the mean absolute activation for each unit.
    """
    ...
```

The implementation must use only NumPy vectorised operations and return a `float64` array. No Python loops are allowed.

## Example

```python
import numpy as np
from rws_activation_based_importance_scoring import score_importance

# activations of shape (3, 4)
acts = np.array([[1, -2, 0, 3],
                 [-1,  5, 2, -4],
                 [ 0, -1, 3,  2]], dtype=np.float64)

imp = score_importance(acts)
print(imp)
# Output: [1.33333333 2.66666667 1.66666667 2.33333333]
```

## What the gate checks

The grader generates a random calibration batch of shape `(64, 128)` and compares your output to a NumPy reference computed as `np.mean(np.abs(activations), axis=0)`. The relative L2 error must satisfy  

$$\mathrm{rel\_err} \le 10^{-6}.$$

A correct implementation will produce an array of type `float64` that matches the reference within this tolerance. A broken solution (e.g., missing the absolute value or using a different dtype) will fail the gate.
