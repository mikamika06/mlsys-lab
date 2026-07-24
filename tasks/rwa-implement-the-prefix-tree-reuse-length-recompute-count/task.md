## Context

A prefix tree (trie) is a data structure that stores a set of strings in a shared‑prefix manner.  
For any query string $s$, the longest prefix that already exists in the trie can be reused; the remaining suffix must be recomputed or inserted.  If $\ell$ denotes this reuse length, then the number of characters that need to be processed anew is

$$\text{recompute} = |s| - \ell.$$

The task is to implement a stream processor that, for each operation, reports both $\ell$ and the recompute count.

## Task

Implement `process_ops(ops)`:

```python
def process_ops(ops: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    ...
```

`ops` is a list of tuples where the first element is either `'add'` or `'query'`.  
For each operation you must return two lists:
* `reuse_lengths`: the reuse length $\ell$ for that operation.
* `recompute_counts`: the recompute count $|s|- \ell$.

The function should maintain an internal prefix tree.  
An `'add'` operation inserts the word into the tree after computing its reuse length; a `'query'` only reports the reuse length and does not modify the tree.

All strings consist of lowercase ASCII letters (`a–z`).  The implementation must run in Python 3.10+ and use only built‑in data structures (no external libraries).

## Example

```python
ops = [
    ('add', 'cat'),
    ('query', 'car'),
    ('add', 'cater'),
    ('query', 'cart')
]
reuse, recompute = process_ops(ops)
print(reuse)      # [0, 2, 3, 3]
print(recompute)  # [3, 1, 2, 1]
```

Explanation:

* `'cat'` is new → reuse $=0$, recompute $=3$.
* `'car'` shares prefix `"ca"` with `'cat'` → reuse $=2$, recompute $=1$.
* `'cater'` extends `'cat'` by `"er"` → reuse $=3$, recompute $=2$.
* `'cart'` shares prefix `"car"` (from the query) but not in the tree until after the previous add, so reuse $=3$, recompute $=1$.

## What the gate checks

The grader generates random sequences of `'add'` and `'query'` operations.  
For each sequence it computes a reference answer by brute‑force scanning all stored words before each operation.  Your implementation must produce exactly the same two lists for every test case; otherwise the `exact_match` gate fails.
