## Context

In many caching systems the *eviction policy* determines which items are kept in memory when new data arrives and capacity is exceeded.  
Three common policies are:

- **Window (LRU)** – keep at most $W$ distinct tokens that have appeared within the last $W$ positions of a stream.  The cache behaves like an LRU stack: the most recently seen token is on top, the least recent at the bottom.

- **Heavy‑hitter** – retain the $K$ most frequent tokens in the *entire* stream.  When frequencies tie, the smaller token value wins.  The output list is sorted by decreasing frequency (and ascending token for ties).

- **Recent‑only (FIFO)** – keep a fixed-size queue of the last $N$ tokens seen, preserving their arrival order.  Duplicates are allowed; the output is the queue from oldest to newest.

All three policies produce an ordered list of tokens that can be compared directly.

## Task

Implement `identify_policy(retained_set: List[int], stream: List[int]) -> str`.

The function receives:

- `retained_set`: the list produced by one of the three eviction policies above.
- `stream`: the original token stream that was processed to produce the set.

It must return a string label identifying the policy that generated `retained_set`.  The allowed labels are `"window"`, `"heavy_hitter"`, and `"recent_only"`.

The implementation should be deterministic, use only standard library modules (`collections` is permitted), and run in $O(n)$ time where $n$ is the length of `stream`.

## Example

```python
import collections

def identify_policy(retained_set, stream):
    # Implementation omitted for brevity
    ...

# Example usage
stream = [2, 3, 2, 5, 3, 4, 1, 2, 6, 3]
retained = [3, 2, 6, 4, 1]          # produced by the window policy (W=5)
label = identify_policy(retained, stream)
print(label)   # → "window"
```

## What the gate checks

The grader generates random streams and produces a retained set from one of the three policies.  
It then calls your `identify_policy` function and verifies that the returned string matches the policy that produced the set.  The check is exact: `"window"`, `"heavy_hitter"` or `"recent_only"`.
