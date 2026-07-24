## Context

NumPy broadcasting compares array dimensions from the rightmost dimension toward the left. If two shapes are compatible, each aligned pair of dimensions must either be equal or one of them must be $1$.

For arrays with shapes

$$A \in \mathbb{R}^{n_1 \times n_2 \times \dots \times n_k}$$

and

$$B \in \mathbb{R}^{m_1 \times m_2 \times \dots \times m_j},$$

NumPy conceptually pads the shorter shape on the left with dimensions of size $1$ and aligns dimensions from the right. A manual implementation that aligns dimensions from the left can produce incorrect shapes or values.

For example, a shape $(2, 3, 4)$ array and a shape $(3, 4)$ array are compatible because $(3, 4)$ is aligned with the last two dimensions of $(2, 3, 4)$.

## Task

Implement `broadcast_add_right(a, b)`:

```python
def broadcast_add_right(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

Return the elementwise sum using right-aligned broadcasting semantics. The output must match NumPy's `a + b` behavior for compatible inputs.

Do not use Python loops to manually expand elements. Use NumPy operations and preserve the numeric values and resulting shape.

## Example

```python
import numpy as np

a = np.ones((2, 3, 4))
b = np.arange(4)

out = broadcast_add_right(a, b)

# out has shape (2, 3, 4)
# out[i, j, k] == a[i, j, k] + b[k]
```

## What the gate checks

The gate compares the returned shape and values against the NumPy reference operation $a + b$ on several arrays with different ranks.

The metric `exact_match` must equal $1.0$. Any left-aligned broadcasting implementation fails because it does not match NumPy's right-aligned broadcasting rules.
