## Context

Memory-efficient attention algorithms (like FlashAttention) compute exact attention by splitting the large keys and values matrices into smaller blocks. For each block, they compute a "partial" attention output, and then merge these partials iteratively or in parallel.

For a specific query row, suppose we have $k$ key-value blocks. For the $i$-th block, the standard softmax requires us to track three local quantities:
1. The local maximum score: $m_i = \max(S_i)$
2. The local sum of exponentials: $l_i = \sum \exp(S_i - m_i)$
3. The local normalized output: $O_i = \frac{\sum \exp(S_i - m_i) V_i}{l_i}$

To merge these $k$ block-level partials into the mathematically exact global attention output $O$, we can use the online softmax trick. The global maximum is $m = \max_i(m_i)$. The block outputs must be re-weighted relative to this global maximum before summing.

## Task

Write `merge_attention_blocks(m_blocks, l_blocks, O_blocks)`:

```python
import numpy as np

def merge_attention_blocks(
    m_blocks: np.ndarray, 
    l_blocks: np.ndarray, 
    O_blocks: np.ndarray
) -> np.ndarray:
    ...
```

Given the partial components for $k$ blocks, combine them to compute the final attention output. 

Input array shapes:
- `m_blocks`: `(num_blocks, seq_len, 1)`
- `l_blocks`: `(num_blocks, seq_len, 1)`
- `O_blocks`: `(num_blocks, seq_len, d_v)`

Return the final combined output `O` of shape `(seq_len, d_v)`.

## Example

```python
import numpy as np

# Suppose we have 2 blocks, sequence length of 3, and d_v of 4
m_blocks = np.random.randn(2, 3, 1)
l_blocks = np.random.uniform(1, 10, size=(2, 3, 1))
O_blocks = np.random.randn(2, 3, 4)

O_final = merge_attention_blocks(m_blocks, l_blocks, O_blocks)
# O_final shape: (3, 4)
```

## What the gate checks

The grader splits a full un-blocked attention computation into several blocks, extracts the local $m_i$, $l_i$, and $O_i$, and passes them to your function. It expects the maximum absolute difference between your combined output and the standard full attention output to be less than $10^{-6}$.
