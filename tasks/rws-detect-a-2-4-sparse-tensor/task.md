## Context

In many compressed neural‑network formats a *2:4 sparse* tensor is one in which the
elements are grouped into blocks of four and each block contains exactly two
non‑zero values.  Formally, for a NumPy array $A$ with shape $(\dots , n)$,
let $n$ be divisible by $4$.  For every index tuple $(i_1,\dots,i_{k-1},j)$
with $0 \le j < n/4$, the sub‑array

$$B = A[i_1,\dots,i_{k-1},\,4j:4j+4]$$

must satisfy $\#\{x \in B : x \neq 0\}=2$.

## Task

Implement `is_2x4_sparse(tensor)`:

```python
def is_2x4_sparse(tensor: np.ndarray) -> bool:
    ...
```

The function receives a NumPy array of arbitrary shape.  It should return
`True` if the last dimension length is a multiple of $4$ and every block of
four consecutive elements along that axis contains exactly two non‑zero
entries; otherwise it returns `False`.  The implementation must use only
NumPy operations – no explicit Python loops.

## Example

```python
import numpy as np
A = np.array([[1, 0, 2, 0,   # block 1: two non‑zeros
               3, 4, 0, 0],  # block 2: two non‑zeros
              [0, 5, 0, 6,
               7, 0, 8, 0]]) # all blocks satisfy the rule

print(is_2x4_sparse(A))
# True

B = np.array([[1, 0, 0, 0],   # block 1: only one non‑zero
              [3, 4, 5, 6]])  # block 2: four non‑zeros

print(is_2x4_sparse(B))
# False
```

## What the gate checks

The grader evaluates your function on a handful of tensors – some that are
valid 2:4 sparse and others that are not.  The metric `exact_match` compares
the boolean you return with the reference implementation; it must be exactly
equal for all test cases.
