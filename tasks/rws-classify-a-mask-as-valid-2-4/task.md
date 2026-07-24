## Context

In many neural‑network pruning schemes a *2:4 sparsity pattern* is used.  
For every contiguous block of four elements along the last dimension, exactly two
entries are allowed to be non‑zero. This guarantees a fixed density while keeping
the mask structure regular enough for efficient hardware execution.

Mathematically, let $m \in \{0,1\}^{\dots\times N}$ with $N$ divisible by $4$.  
Define groups

$$g_i = m[..., 4i:4i+4] , \qquad i=0,\dots,\frac{N}{4}-1.$$

The mask is *valid* iff for every group

$$\sum_{j=0}^{3} g_i[j] = 2 .$$

## Task

Implement the function

```python
def classify_mask_2_4(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    ...
```

`mask` is a binary NumPy array of arbitrary shape whose last dimension length is a multiple of four.  
The function must return:

* `group_validity`: a boolean array with the same leading dimensions as `mask` and
  a final size of $N/4$, where each entry indicates whether the corresponding group satisfies the 2:4 rule.
* `overall`: a single Python `bool` that is `True` iff *all* groups are valid.

The implementation must be fully vectorised; no explicit Python loops are allowed.  
Use only NumPy operations and standard library functions.

## Example

```python
import numpy as np
mask = np.array([[1, 0, 1, 0,   # group 0: sum=2 -> valid
                  0, 1, 0, 1],  # group 1: sum=2 -> valid
                 [1, 1, 0, 0,   # group 0: sum=2 -> valid
                  1, 0, 1, 0]]) # group 1: sum=3 -> invalid

group_validity, overall = classify_mask_2_4(mask)
print(group_validity)  # [[ True,  True],
                       #  [ True, False]]
print(overall)         # False
```

## What the gate checks

The grader computes a reference answer using NumPy and compares it to the candidate’s output with an exact match.  
If both `group_validity` arrays are element‑wise equal **and** the `overall` booleans match, the solution passes.
