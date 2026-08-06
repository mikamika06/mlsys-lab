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
def merge_attention_blocks(
    m_blocks: list[list[list[float]]],
    l_blocks: list[list[list[float]]],
    O_blocks: list[list[list[float]]]
) -> list[list[float]]:
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
# Suppose we have 2 blocks, sequence length of 3, and d_v of 4
m_blocks = [
    [[0.1], [0.2], [0.3]],
    [[0.4], [0.5], [0.6]]
]
l_blocks = [
    [[1.0], [2.0], [3.0]],
    [[4.0], [5.0], [6.0]]
]
O_blocks = [
    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
    [[1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4]]
]

O_final = merge_attention_blocks(m_blocks, l_blocks, O_blocks)
# O_final shape: (3, 4)
```

## What the gate checks

The grader splits a full un-blocked attention computation into several blocks, extracts the local $m_i$, $l_i$, and $O_i$, and passes them to your function. It expects the maximum absolute difference between your combined output and the standard full attention output to be less than $10^{-6}$.
