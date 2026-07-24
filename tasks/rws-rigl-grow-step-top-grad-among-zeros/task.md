## Context

In sparse neural‑network training the **RigL** algorithm periodically *grows* a small number of previously inactive connections.  
Let $g \in \mathbb{R}^n$ be the gradient vector for all parameters and let
$\mathbf m\in\{0,1\}^n$ be the binary mask that indicates which weights are currently active ($m_i=1$) or dormant ($m_i=0$).  
During a grow step we select the $k$ dormant indices with the largest absolute gradient magnitude and set their mask entry to one.  The corresponding weight is then initialized to zero.

Mathematically, if
$$\mathcal Z = \{\,i : m_i = 0\,\}$$
and $k\le |\mathcal Z|$, we choose
$$S = \arg\max_{J\subseteq \mathcal Z,\; |J|=k}\;\sum_{j\in J} |g_j|,$$
and return the new mask $\mathbf m'=\mathbf m + \mathbf 1_S$.

## Task

Implement `rigl_grow_step`:

```python
def rigl_grow_step(gradients: np.ndarray,
                   mask: np.ndarray,
                   grow_count: int) -> np.ndarray:
    ...
```

* `gradients`: a one‑dimensional NumPy array of shape `(n,)`.
* `mask`: a boolean or integer array of the same shape indicating active weights.
* `grow_count`: how many dormant connections to activate.

The function must return a new mask of the same shape and dtype as `mask`.  
It should **not** modify its inputs.  The returned mask must have exactly
`grow_count` additional ones compared with the input, placed at the indices
with the largest $|g_i|$ among those where `mask[i]==0`.

## Example

```python
import numpy as np
grad = np.array([0.1, -2.5, 3.0, 0.4])
mask = np.array([1, 0, 0, 1], dtype=bool)
new_mask = rigl_grow_step(grad, mask, grow_count=1)
print(new_mask)   # [ True False  True  True]
```

The only dormant index with the largest absolute gradient is `2` (|3.0|).

## What the gate checks

Two checks are performed:

* **Exact match** – The returned mask must be *exactly* equal to a reference
  implementation that uses NumPy’s vectorised operations.
* **Shape & dtype** – The output must have the same shape and boolean dtype as
  the input mask.

If either check fails, the solution is rejected.===== FILE: gen_fixtures.py =====
"""No external fixtures; test cases are generated in the grader."""
