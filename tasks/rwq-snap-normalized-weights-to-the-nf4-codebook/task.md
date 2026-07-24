## Context

The NF4 quantization scheme maps a real‑valued weight in the interval $[-1,\,1]$ to one of sixteen fixed levels that approximate a standard normal distribution truncated at $\pm 1$. The canonical codebook used by production libraries is

$$
\mathcal{L}=\{-1.00000000,\,-0.93333333,\,-0.80000000,\,-0.66666667,\,
-0.53333333,\,-0.40000000,\,-0.26666667,\,-0.13333333,\\
\phantom{\mathcal{L}=}\;0.00000000,\;0.13333333,\;0.26666667,\;
0.40000000,\;0.53333333,\;0.66666667,\;0.80000000,\;0.93333333\}.
$$

Given a weight $w$, the nearest‑neighbour assignment chooses the index
$i$ that minimises $\lvert w-\mathcal{L}_i\rvert$.  The resulting integer
is stored as an unsigned 8‑bit value, because only sixteen distinct values are needed.

## Task

Implement the function `snap_nf4(weights)`:

```python
def snap_nf4(weights: np.ndarray) -> np.ndarray:
    ...
```

`weights` is a one‑dimensional NumPy array of type `float32` or `float64`
containing values already normalised to the interval $[-1,\,1]$.
The function must return an array of the same shape containing
the unsigned 8‑bit indices of the nearest NF4 level for each element.

The implementation has to be fully vectorised; no explicit Python loops are allowed.

## Example

```python
import numpy as np
from your_module import snap_nf4

weights = np.array([-1.0, -0.5, 0.0, 0.7, 0.93333333])
indices = snap_nf4(weights)
print(indices)          # [0 6 8 13 15]
```

## What the gate checks

The grader computes a reference assignment using the exact NF4 codebook
and compares it to your output with an element‑wise equality test.
A solution passes if and only if all indices match exactly.

No timing or memory constraints are imposed; the focus is correctness.
