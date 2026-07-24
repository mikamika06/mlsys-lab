## Context

A NumPy array is a pointer plus a `shape` tuple and a `strides` tuple. Element $(i_0,\dots,i_{k-1})$
lives at byte offset

$$
\text{offset} = \sum_{d=0}^{k-1} i_d \cdot \text{strides}[d].
$$

Broadcasting is not a copy: it is this formula with a **zero stride**. If $\text{strides}[d] = 0$,
then index $i_d$ contributes nothing to the offset, so every position along axis $d$ reads the *same*
memory. That is exactly how `np.broadcast_to` turns a $(1, 5)$ array into a $(3, 5)$ array while
touching zero extra bytes.

The broadcasting rule: right-align the two shapes, left-padding the source with $1$s. For each
aligned pair $(\text{dim}, \text{tgt})$,

$$
\text{strides}_{\text{out}}[d] =
\begin{cases}
\text{strides}_{\text{src}}[d], & \text{dim} = \text{tgt} > 1,\\[2pt]
0, & \text{dim} = 1,\\[2pt]
\text{error}, & \text{otherwise.}
\end{cases}
$$

Padded (missing) leading axes count as $\text{dim} = 1$, hence stride $0$.

## Task

Implement `broadcast_to_strided` in `solve.py`:

```python
from numpy.lib.stride_tricks import as_strided

def broadcast_to_strided(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    ...
```

Requirements:

1. The result must equal `np.broadcast_to(a, shape)` element for element, with the same `dtype`.
2. It must be a **zero-copy view**: `np.shares_memory(result, a)` is `True`, and the result's
   `strides` must match NumPy's own broadcast strides (i.e. $0$ on every broadcast axis).
   Build it with `as_strided(a, shape=..., strides=..., writeable=False)`.
3. If `a.shape` cannot be broadcast to `shape` — a mismatched non-1 dimension, or `len(shape)`
   smaller than `a.ndim` — raise `ValueError`.

`np.broadcast_to`, `np.broadcast_arrays`, `np.tile`, `np.repeat` and `np.resize` are blocked by the
grader. The input may itself be non-contiguous, so derive the output strides from `a.strides`, not
from `a.shape` alone.

## Example

```python
import numpy as np

a = np.arange(3.0)                  # shape (3,),   strides (8,)
v = broadcast_to_strided(a, (4, 3))

v.shape     # (4, 3)
v.strides   # (0, 8)   <- axis 0 is free
v.base is a.base or np.shares_memory(v, a)   # True, no data copied
v
# array([[0., 1., 2.],
#        [0., 1., 2.],
#        [0., 1., 2.],
#        [0., 1., 2.]])
```

## What the gate checks

Eight cases (scalars, leading-axis padding, size-1 middle axes, a non-contiguous source, an `int32`
source, and an identity broadcast) are compared against the live `np.broadcast_to` oracle:

* `byte_exact_fraction` — fraction of cases whose materialised bytes match the oracle exactly; must
  be $1.0$.
* `zero_copy_view_fraction` — fraction of cases where the result shares memory with `a` **and** has
  the oracle's exact `strides`; must be $1.0$. Materialising a copy fails here even if the values
  are right.
* `raises_on_bad_shape_fraction` — fraction of three invalid requests that raise `ValueError`; must
  be $1.0$.
