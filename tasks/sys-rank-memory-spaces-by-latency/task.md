## Context

In a typical GPU memory hierarchy the access time increases from registers to shared memory, then to L2 cache and finally global memory. We can model each memory space by a representative Python object whose allocation size is correlated with its access latency in CPython: small integers are stored in a fast internal pool, lists require more bookkeeping, NumPy arrays involve contiguous C buffers, and dictionaries have the largest overhead.

Let $L_{\text{reg}}, L_{\text{sh}}, L_{\text{L2}}, L_{\text{glb}}$ denote the relative latency of each space. We want to return a list sorted from smallest to largest latency.

## Task

Implement `rank_memory_spaces()`:

```python
def rank_memory_spaces() -> List[str]:
    ...
```

It should return a list of the four memory‑space names in order of increasing access latency: `'register'`, `'shared'`, `'L2'`, `'global'`. The implementation must use only standard library modules (`sys`, `numpy`) and no hard‑coded ordering.

## Example

```python
>>> rank_memory_spaces()
['register', 'shared', 'L2', 'global']
```

## What the gate checks

The grader computes a reference ordering by measuring `sys.getsizeof` on representative objects for each memory space. The candidate solution must return exactly that list; otherwise the `exact_match` metric fails.
