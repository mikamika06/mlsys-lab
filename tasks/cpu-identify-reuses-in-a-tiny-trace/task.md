## Context

Caches are a key component of modern CPUs, providing fast access to recently used data.  
A cache is organized into *sets* and each set contains a number of *ways*.  
When an address is accessed it maps to a particular set; if the line containing that
address is already present in one of the ways of that set, the access is a **hit**,
otherwise it is a **miss**.  A common replacement policy is Least‑Recently‑Used (LRU).

The *reuse distance* of an access is the number of distinct addresses accessed since
the previous reference to the same address.  Small reuse distances lead to cache hits.

In this task you will be given a fixed trace of byte addresses and asked to count how many
of those accesses are cache hits for a deterministic LRU cache with known parameters.

## Task

Implement the function `count_cache_hits(trace)` that takes a one‑dimensional NumPy array
`trace` of integer byte addresses (dtype `int64`) and returns an integer equal to the
number of cache hits when this trace is processed by an LRU cache with the following
configuration:

* line size: 8 bytes  
* number of sets: 4  
* associativity (ways): 2  

You may use the provided `cachesim.simulate` helper from the Arena runtime, which
accepts the same arguments and returns a dictionary containing at least the keys
`'hits'` and `'misses'`.

```python
def count_cache_hits(trace: np.ndarray) -> int:
    ...
```

The function must return an `int`.  It should not print anything.

## Example

```python
import numpy as np

# A tiny trace of byte addresses (each address is the start of a cache line)
trace = np.array([0, 8, 16, 24, 32, 40, 48, 56,
                  0, 8, 16, 24], dtype=np.int64)

hits = count_cache_hits(trace)
print(hits)   # → 4
```

In this example the first eight accesses are all misses; the last four accesses
re‑reference addresses that are still present in the cache, yielding four hits.

## What the gate checks

The grader runs your implementation on a fixed trace (the one shown above) and
compares the returned integer against the reference value computed by the same
`cachesim.simulate` call.  The metric is `exact_match`; you must return exactly
the correct number of hits.
