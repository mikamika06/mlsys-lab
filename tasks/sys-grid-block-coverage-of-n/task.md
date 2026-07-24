## Context

When processing a sequence of $N$ items in parallel, it is common to split the data into fixed‑size blocks (or tiles).  
If each block can hold at most $B$ elements, then the number of blocks needed is  

$$
G = \left\lceil \frac{N}{B} \right\rceil .
$$

The last block may be only partially filled.  For many low‑level kernels we need an *active mask* that tells which positions inside each block actually contain data.

## Task

Implement `block_coverage(n: int, block_size: int) -> Tuple[int, np.ndarray]`:

```python
def block_coverage(n: int, block_size: int):
    ...
```

It must return a tuple `(grid_size, mask)` where  

* `grid_size` is the integer $G$ defined above.  
* `mask` is a boolean NumPy array of shape `(grid_size, block_size)`.  For every index $i \in [0,N)$ the element `mask[i // block_size, i % block_size]` must be `True`; all other entries are `False`.

The implementation should use only NumPy operations; no explicit Python loops over the elements.

## Example

```python
import numpy as np
grid_size, mask = block_coverage(10, 4)
print(grid_size)          # 3
print(mask.astype(int))
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 0 0 0]]
```

## What the gate checks

The grader computes a reference solution with NumPy and compares it to your output using an `exact_match` metric.  Your implementation must produce exactly the same integer grid size and boolean mask as the reference for all test cases.
