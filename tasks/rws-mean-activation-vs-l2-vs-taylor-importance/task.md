## Context

In many neural‑network libraries the importance of a module or parameter
is estimated from its activations and gradients.  
Three common signals are:
- The mean absolute activation $\lVert a\rVert_{1}/d$,
- The Euclidean norm $\lVert a\rVert_{2}$, and
- A first–order Taylor estimate $|g\,w|$, where $g$ is the gradient of the loss
  with respect to the module’s output and $w$ is its weight.

These signals are used for pruning, quantisation or other model‑compression
techniques.  
The task below asks you to compute all three scores for a collection of
structures.

## Task

Implement `importance_scores(structures)`:

```python
def importance_scores(
    structures: dict[int, tuple[np.ndarray, np.ndarray, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

`structures` maps an integer identifier to a tuple `(activations,
gradients, weight)`.  
Return three 1‑D NumPy arrays of type `float64`, each containing the
corresponding score for every structure in ascending key order.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
structures = {
    0: (rng.standard_normal(4), rng.standard_normal(4), 0.5),
    1: (rng.standard_normal(3), rng.standard_normal(3), -1.2),
}
mean_abs, l2_norm, taylor_imp = importance_scores(structures)

# mean_abs   ≈ array([...])
# l2_norm    ≈ array([...])
# taylor_imp ≈ array([...])
```

(The exact numeric values depend on the random seed.)

## What the gate checks

The grader computes a reference implementation using NumPy and compares
your three output arrays element‑wise.  All three must match exactly
(`np.allclose(..., rtol=0, atol=0)`).  Any deviation causes the task to fail.
