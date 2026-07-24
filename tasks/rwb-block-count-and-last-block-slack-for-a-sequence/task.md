## Context

In many data‑processing pipelines a sequence of length $L$ is split into contiguous blocks of fixed size $B$.  
The number of blocks needed to cover the whole sequence is given by the ceiling division

$$n_{\text{blocks}} = \left\lceil \frac{L}{B} \right\rceil
   = \frac{L + B - 1}{B}\;\;(\text{integer division}).$$

Because $L$ may not be an exact multiple of $B$, the last block is usually partially empty.  
The amount of unused slots in that final block – often called *slack* or *wasted space* – is

$$\text{slack} = n_{\text{blocks}}\cdot B - L.$$

Both quantities are integers and can be computed efficiently with vectorised NumPy operations.

## Task

Implement the function `block_stats` that, given an array of sequence lengths and a block size, returns two integer arrays:

```python
def block_stats(seqlens: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `seqlens`: 1‑D NumPy array of non‑negative integers.  
* `block_size`: positive integer.

The function should return a tuple `(num_blocks, slack)` where each element is a NumPy array of the same shape as `seqlens` and dtype `int64`.  
No Python loops are allowed; use vectorised arithmetic only.

## Example

```python
import numpy as np
from your_module import block_stats

seqlens = np.array([3, 8, 15])
block_size = 4

num_blocks, slack = block_stats(seqlens, block_size)
print(num_blocks)   # [1 2 4]
print(slack)        # [1 0 1]
```

Explanation:  
* For length 3 with block size 4 we need one block and waste $4-3=1$ slot.  
* Length 8 fits exactly into two blocks, so no slack.  
* Length 15 requires four blocks ($\lceil 15/4\rceil = 4$) and wastes $16-15=1$ slot.

## What the gate checks

The grader computes a reference implementation using NumPy’s integer arithmetic:

```python
ref_blocks = (seqlens + block_size - 1)//block_size
ref_slack  = ref_blocks*block_size - seqlens
```

It then compares your output to these arrays element‑wise.  
If both returned arrays match exactly, the metric `exact_match` is set to 1.0; otherwise it is 0.0.  
The task has a single gate that requires `exact_match == 1.0`.
