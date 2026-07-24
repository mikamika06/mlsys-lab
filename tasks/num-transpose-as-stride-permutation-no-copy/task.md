## Context

A NumPy array stores values in a buffer together with metadata describing how to
interpret that buffer. For an array with shape $(s_0, s_1, \dots, s_{k-1})$,
the stride tuple $(t_0, t_1, \dots, t_{k-1})$ gives the byte offset added when
an index increases by one along each axis.

A transpose does not need to move data. It can be represented by permuting the
shape and stride metadata:

$$
\mathrm{shape}' = (s_{p_0}, s_{p_1}, \dots, s_{p_{k-1}})
$$

and

$$
\mathrm{strides}' = (t_{p_0}, t_{p_1}, \dots, t_{p_{k-1}}),
$$

where $p$ is the axis permutation. A correct implementation returns a view that
shares the original storage while presenting elements in the transposed order.

Copying the data would allocate a new buffer and break the no-copy requirement.
For large arrays this distinction affects both memory usage and performance.

## Task

Implement `transpose_view(A, axes)`:

```python
def transpose_view(A: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    ...
```

Return a transposed view of `A` using the given axis permutation. The returned
array must:

- have the same values as `np.transpose(A, axes=axes)`,
- share memory with `A`,
- avoid copying the underlying data,
- preserve the dtype of `A`.

Do not materialize a copied transpose.

## Example

```python
import numpy as np

A = np.arange(12, dtype=np.int64).reshape(3, 4)
B = transpose_view(A, (1, 0))

# B has shape (4, 3)
# B[2, 1] == A[1, 2]
# np.shares_memory(A, B) is True
```

## What the gate checks

The gate uses NumPy's transpose implementation as the reference oracle. It
materializes the candidate and reference arrays and compares their byte content
with `byte_exact_fraction`.

The gate also verifies that the returned array shares memory with the input
array. Returning a copied transpose fails even when the values are correct.
