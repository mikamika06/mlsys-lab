## Context

In a paged KV‑cache used by transformer models, each sequence of tokens is stored in contiguous blocks of fixed size \(B\). If a sequence has length \(\ell\), the number of blocks required is
$$
b = \left\lceil \frac{\ell}{B} \right\rceil .
$$
The cache therefore contains \(bB\) slots, of which only \(\ell\) are used. The remaining
\(w = bB - \ell\) slots are wasted.

## Task

Implement `compute_block_and_waste(block_size: int, seq_lengths: list[int]) -> tuple[list[int], list[int]]` that returns two lists:

* `blocks`: the number of blocks allocated for each sequence.
* `wasted`: the number of unused slots in those blocks.

Both outputs must be Python `list`s of integers and have the same length as `seq_lengths`.

## Example

```python
block_size = 4
seq_lengths = [5, 8, 3]
blocks, wasted = compute_block_and_waste(block_size, seq_lengths)
print(blocks)   # [2, 2, 1]
print(wasted)   # [3, 0, 1]
```

## What the gate checks

The grader computes the reference answer using NumPy integer arithmetic and compares it to your output with an exact match. Any mismatch causes the task to fail.
