## Context

Sequence parallelism divides a long sequence into chunks processed by multiple ranks. In causal attention, a chunk near the end of the sequence has more work because it can attend to all previous chunks.

For $n$ sequence chunks, chunk $i$ has causal work weight

$$
w_i = i + 1
$$

because it attends to chunks $0, 1, \dots, i$. The total causal work is

$$
W = \sum_{i=0}^{n-1} w_i .
$$

A balanced assignment tries to distribute these weights across $r$ ranks so that no rank receives much more causal work than another. A greedy zig-zag strategy first considers chunks from largest causal load to smallest, then assigns each chunk to the currently lightest rank. When multiple ranks have the same current load, the tie is broken by the zig-zag rank order.

## Task

Implement `zig_zag_causal_assignment(num_chunks, num_ranks)`:

```python
def zig_zag_causal_assignment(num_chunks: int, num_ranks: int) -> list[int]:
    ...
```

Return a list of length `num_chunks`. The value at index `i` is the rank that should process chunk `i`.

The assignment algorithm is:

1. Compute causal weights $w_i = i + 1$.
2. Visit chunks in descending weight order.
3. Maintain each rank's accumulated work.
4. For each chunk, choose the rank with the smallest accumulated work. If several ranks are tied, use the current zig-zag tie order.
5. After each assignment, reverse the tie order direction.

Ranks are numbered from $0$ to `num_ranks - 1`.

The function should return only the rank list. It should not return work totals.

## Example

```python
assignment = zig_zag_causal_assignment(5, 2)
print(assignment)
```

One valid output from the specified algorithm is:

```text
[0, 1, 0, 1, 1]
```

The returned value maps chunks to ranks. Chunk `4` is assigned first because it has the largest causal work.

## What the gate checks

The gate computes the reference assignment with the same causal balancing algorithm and compares the returned list exactly.

The oracle verifies the complete placement behavior, including descending causal loads, least-loaded rank selection, and zig-zag tie handling. A solution that only assigns chunks round-robin or balances the number of chunks instead of the causal work will fail.
