## Context

In systems that manage fixed-size block pools — GPU memory allocators, network
buffer managers, ring-buffer caches — variable-length data is stored by
allocating an integer number of blocks from a shared pool.  A sequence of
length $\ell$ occupying a block of size $s$ needs

$$b(\ell) = \left\lceil \frac{\ell}{s} \right\rceil$$

blocks.  The block pool holds $B$ blocks in total.  Given $N$ sequences with
lengths $\ell_1, \ell_2, \ldots, \ell_N$, we want to maximise the number of
sequences admitted concurrently so that total block usage does not exceed $B$.

Choosing which sequences to admit is a classic count-maximisation problem.
Because every sequence we admit consumes at least $b_{\min}$ blocks and at most
$b_{\max}$ blocks, admitting shorter sequences first leaves more room for
additional sequences.  Formally, sort the per-sequence block requirements in
non-decreasing order:

$$b_{(1)} \leq b_{(2)} \leq \cdots \leq b_{(N)}$$

then greedily admit $b_{(1)}, b_{(2)}, \ldots$ until the cumulative sum
$\sum_{i=1}^{k} b_{(i)}$ first exceeds $B$.  The answer is $k-1$ (or $N$ if
all sequences fit).

This greedy strategy is optimal for maximising admission count: any selection
of $k$ sequences that does not pick the $k$ shortest will use at least as many
blocks (by an exchange argument), so it cannot admit more.

## Task

Implement:

```python
def max_concurrent_sequences(lengths: list[int], block_size: int, total_blocks: int) -> int:
    ...
```

**Parameters**

- `lengths` — a list of $N$ non-negative integers, the sequence lengths.
- `block_size` — a positive integer $s \geq 1$, the number of elements each block holds.
- `total_blocks` — a non-negative integer $B$, the pool capacity in blocks.

**Returns**

The maximum number of sequences that can be admitted concurrently without
exceeding the block pool.  Sequences are admitted greedily by fewest blocks
needed first.

You may use only the standard library (no NumPy required).  All arithmetic is
integer-based.

## Example

```python
# Pool has 5 blocks, each block holds 3 elements.
# Sequences of lengths [1, 2, 3, 4, 5] need [1, 1, 1, 2, 2] blocks.
# Sorted: [1, 1, 1, 2, 2].  Greedy: 1+1+1+2 = 5 ≤ 5 → 4 admitted.
max_concurrent_sequences([1, 2, 3, 4, 5], block_size=3, total_blocks=5)
# → 4
```

Another example:

```python
# Three sequences of length 10, block_size=10, pool=2.
# Each needs ceil(10/10) = 1 block.  Can fit 2 before pool is full.
max_concurrent_sequences([10, 10, 10], block_size=10, total_blocks=2)
# → 2
```

## What the gate checks

The gate uses the metric `exact_match`.  An oracle in `check.py` independently
computes the greedy admission count for every test case (deterministic
pseudo-random lengths and pool sizes seeded at 42, plus hand-picked edge
cases).  Your function's return value must equal the oracle's count for every
case.  Any discrepancy — returning too few (sub-optimal) or too many (exceeding
pool) — is a failure.
