## Context

Magnitude pruning removes a fraction of the smallest‑by‑absolute‑value weights from a neural network.  
Given a weight vector $w \in \mathbb{R}^n$ and a keep fraction $\alpha\in(0,1)$, we wish to retain exactly $\lceil \alpha n\rceil$ weights with largest magnitude.

When several weights share the same absolute value as the cutoff, the choice of which ones to keep must be deterministic.  The standard convention is **stable tie‑breaking**: after sorting by descending $|w_i|$, ties are broken by the original index order (i.e., the first occurrence in the array wins).  This guarantees reproducible masks across runs and implementations.

## Task

Implement `magnitude_prune_mask`:

```python
def magnitude_prune_mask(weights: np.ndarray, keep_fraction: float) -> np.ndarray:
    ...
```

* `weights` is a one‑dimensional NumPy array of arbitrary dtype.
* `keep_fraction` specifies the fraction of weights to retain; it satisfies $0<\alpha<1$.
* The function must return a boolean mask of length `len(weights)` where `True` indicates that the corresponding weight is kept.

The implementation must use only NumPy operations (no Python loops) and be fully deterministic.  It should handle edge cases such as zero weights, duplicate magnitudes, and very small arrays.

## Example

```python
import numpy as np
w = np.array([0.5, -1.2, 0.3, -1.2, 0.8])
mask = magnitude_prune_mask(w, keep_fraction=0.6)
# mask == array([False, True, False, False, True])  # keeps two largest magnitudes
```

## What the gate checks

The grader computes an **oracle** by performing a stable sort of `|weights|` in descending order and selecting the first $\lceil \alpha n\rceil$ indices.  
Your mask must match this oracle exactly; any deviation (including incorrect tie handling or wrong threshold comparison) causes the gate to fail.
