## Context

In many sequence‑processing libraries a *packed batch* is represented by a single flat array of tokens together with an offset array that marks the start of each original sequence.  
If we denote the lengths of the $N$ sequences by $\ell_1,\dots,\ell_N$, the corresponding **cu_seqlens** (cumulative‑sum offsets) are defined as

$$
\text{cu\_seqlens}_0 = 0, \qquad
\text{cu\_seqlens}_{i} = \sum_{j=1}^{i}\ell_j,\; i=1,\dots,N.
$$

Thus $\text{cu\_seqlens}$ has length $N+1$ and the slice of tokens belonging to sequence $i$ is
$\text{tokens}[\text{cu\_seqlens}_{i}:\text{cu\_seqlens}_{i+1}]$.

## Task

Implement `build_cu_seqlens`:

```python
def build_cu_seqlens(lengths: np.ndarray) -> np.ndarray:
    ...
```

The function receives a one‑dimensional NumPy array of integer sequence lengths (dtype `int32`) and must return the cumulative‑sum offset array described above, also as an `int32` NumPy array. No Python loops are allowed; use vectorised NumPy operations only.

## Example

```python
import numpy as np
lengths = np.array([3, 2, 4], dtype=np.int32)
cu_seqlens = build_cu_seqlens(lengths)
print(cu_seqlens)          # [0 3 5 9]
```

The returned array has shape `(len(lengths)+1,)` and contains the offsets `0, 3, 5, 9`.

## What the gate checks

A single gate named **exact_match** verifies that the output of your implementation is exactly equal to a NumPy reference computed as
$$\text{np.concatenate}([[0], \text{np.cumsum(lengths)]})$$
for several test cases. The comparison is performed on Python lists, so any difference in values or ordering will cause the gate to fail.
