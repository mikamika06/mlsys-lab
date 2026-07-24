## Context

In many machine‑learning pipelines a batch of rows is annotated with an *adapter id* that indicates which sub‑model or processing path should handle the row.  
For efficient execution it is common to **group** consecutive rows that share the same adapter id, so that each segment can be processed in bulk.

Let $A \in \mathbb{Z}^{n}$ denote the array of adapter ids for a batch of $n$ rows.  
We want to produce

1. A permutation $\pi \in \{0,\dots,n-1\}^n$ such that
   $$A_{\pi[0]} \leq A_{\pi[1]} \leq \dots \leq A_{\pi[n-1]}$$
   and the ordering is **stable**: rows with equal ids keep their original relative order.

2. An array of segment start offsets $S \in \{0,\dots,n\}^{k+1}$ where $k$ is the number of distinct adapter ids in $A$.  
   The first element is always $0$, the last element is $n$, and for each distinct id $u_j$ (sorted ascending) we have
   $$S[j] = \#\{i : A_{\pi[i]} < u_j\}.$$

The pair $(\pi,S)$ allows a downstream routine to iterate over segments by slicing:
```python
for start, end in zip(S[:-1], S[1:]):
    process_rows(pi[start:end])
```

## Task

Implement the function `group_rows_by_adapter`:

```python
import numpy as np
from typing import Tuple

def group_rows_by_adapter(adapter_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    adapter_ids : np.ndarray of shape (n,) and dtype integer
        The adapter id for each row in the batch.

    Returns
    -------
    perm : np.ndarray of shape (n,)
        A stable permutation that sorts `adapter_ids` ascending.
    offsets : np.ndarray of shape (k+1,)
        Segment start indices, where k is the number of unique ids.
        offsets[0] == 0 and offsets[-1] == n.
    """
    ...
```

The implementation must use only NumPy operations; no Python loops are allowed.  
The returned arrays should be of type `np.int64`.

## Example

```python
import numpy as np
ids = np.array([2, 0, 1, 2, 1])
perm, offsets = group_rows_by_adapter(ids)
print(perm)     # [1, 2, 4, 0, 3]
print(offsets)  # [0, 1, 3, 5]
```

Explanation:  
- After stable sorting by id we obtain the order `[0, 1, 1, 2, 2]` corresponding to original indices `[1, 2, 4, 0, 3]`.  
- The unique ids are `[0, 1, 2]`; the counts are `[1, 2, 2]`, so the cumulative offsets are `[0, 1, 3, 5]`.

## What the gate checks

The grader verifies that both returned arrays match a NumPy reference implementation exactly.  
No other metrics are required; a single `exact_match` gate ensures correctness of the permutation and segment boundaries.
