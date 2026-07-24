## Context

In a memory‑allocation system each free block has a size measured in megabytes (MB). When a request for `max_split_size_mb` bytes arrives, the allocator may split an oversized free block into two parts: one of exactly `max_split_size_mb` and a remainder. A block is considered *splittable* if it can be used as‑is (size ≤ max_split_size_mb) **or** if it can be split so that both resulting blocks are usable, i.e.

$$
\text{remainder} = \text{size} - \text{max\_split\_size\_mb}\geq \text{min\_remainder\_mb}.
$$

The minimum remainder size is typically one megabyte; a remainder smaller than this would be discarded as waste.

## Task

Implement `classify_blocks`:

```python
def classify_blocks(sizes: np.ndarray,
                    max_split_size_mb: float,
                    min_remainder_mb: float = 1.0) -> np.ndarray:
    ...
```

The function receives a one‑dimensional NumPy array of free block sizes and returns a boolean array of the same shape where `True` indicates that the corresponding block is splittable under the rule above.

You must use vectorised NumPy operations only; no Python loops are allowed. The result should be of dtype `bool`.

## Example

```python
import numpy as np
sizes = np.array([0.5, 1.2, 3.4, 5.0])
max_split_size_mb = 2.0
D = classify_blocks(sizes, max_split_size_mb)
# D -> array([ True,  True,  True, False], dtype=bool)
```

Explanation:  
- 0.5 MB ≤ 2.0 MB → splittable.  
- 1.2 MB ≤ 2.0 MB → splittable.  
- 3.4 MB > 2.0 MB, remainder = 1.4 MB ≥ 1.0 MB → splittable.  
- 5.0 MB > 2.0 MB, remainder = 3.0 MB ≥ 1.0 MB → splittable (but if we had a stricter min_remainder it could be unsplittable).  

## What the gate checks

The grader computes an oracle using NumPy that applies the same rule to a set of random test cases and compares your output with the oracle via `np.array_equal`. The metric `exact_match` must equal 1.0; any mismatch or exception yields 0.0.
