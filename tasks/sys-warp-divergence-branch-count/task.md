## Context

In GPU programming a *warp* is a group of threads that execute in lock‑step. When the threads in a warp encounter a conditional branch whose predicates differ, the hardware serialises each distinct execution path; this phenomenon is called **branch divergence**. The number of serialized paths for a given warp equals the number of distinct predicate values taken by its constituent threads.

For a 1‑D array `preds` containing an integer identifier for the branch that each thread takes (e.g. 0 or 1), and a fixed `warp_size`, we want to compute, for every consecutive block of `warp_size` elements, how many distinct identifiers appear in that block.

## Task

Implement `warp_divergence_branch_count(preds, warp_size=32)`:

```python
def warp_divergence_branch_count(preds: np.ndarray, warp_size: int = 32) -> np.ndarray:
    ...
```

* `preds` – a one‑dimensional NumPy array of integers (or booleans).  
* `warp_size` – the number of threads per warp; defaults to 32.  

The function must return a one‑dimensional integer array whose length is
`len(preds) // warp_size`. Each element contains the count of distinct values
present in the corresponding block of size `warp_size`.

The implementation should be fully vectorised (no explicit Python loops over
threads). The result type must be an integer NumPy array (`dtype=int64`).

## Example

```python
import numpy as np
preds = np.array([0, 1] * 16)          # 32 threads: alternating branches
counts = warp_divergence_branch_count(preds)
print(counts)                          # [2]

preds = np.array([0]*64 + [1]*64)      # two warps, first all 0, second all 1
print(warp_divergence_branch_count(preds))
# [1, 1]
```

## What the gate checks

The grader computes a reference answer using NumPy’s `unique` on each warp block.
Your implementation must match that reference exactly (`exact_match == 1.0`).  
If the input length is not a multiple of `warp_size`, or if `preds` is not
one‑dimensional, an exception should be raised.
