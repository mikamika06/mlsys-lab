## Context

Streaming large language models often avoid storing every key-value (KV) cache entry
by keeping a fixed set of important positions. A common sparse attention policy keeps
two regions:

- sink tokens: the first $s$ positions, which remain available throughout decoding
- recent window tokens: the last $w$ positions of the sequence

For a sequence of length $n$, the retained set after eviction is

$$
R = \{0,1,\dots,\min(s,n)-1\} \cup
\{\max(0,n-w),\dots,n-1\}.
$$

The final retained KV indices are the sorted unique elements of $R$. The uniqueness
step matters because the sink region and the recent window can overlap when
$s+w > n$.

## Task

Implement `retained_kv_indices(n, s, w)`:

```python
def retained_kv_indices(n: int, s: int, w: int) -> list[int]:
    ...
```

The function receives a sequence length $n$, a sink token count $s$, and a recent
window size $w$. Return the sorted list of integer indices that remain in the KV
cache after applying the sink-plus-window policy.

Assume $n \ge 0$, $s \ge 0$, and $w \ge 0$.

## Example

```python
retained_kv_indices(10, 2, 4)
# [0, 1, 6, 7, 8, 9]

retained_kv_indices(5, 4, 3)
# [0, 1, 2, 3, 4]
```

## What the gate checks

The gate computes the expected retained indices with an independent reference
implementation of the sink-plus-window rule and compares the submitted function
with exact list equality over several sequence lengths and overlap cases.

A solution passes only if it returns the correct sorted unique retained index set.
