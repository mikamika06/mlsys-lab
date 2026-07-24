## Context

In a caching allocator each allocation reserves memory that may be kept for future reuse.
Let a trace of operations consist of allocations and frees.  
For an allocation of size $s$ the allocator **reserves** $s$ bytes; when the object is freed it remains reserved but no longer counts as *allocated*.

Define

- $\text{reserved}(t)$ – total number of bytes that have ever been requested up to time $t$.
- $\text{allocated}(t)$ – sum of sizes of all objects still live at time $t$.

The **fragmentation gap** is the difference between the peak reserved amount and the maximum concurrent allocation:

$$
\text{gap} \;=\; \max_t \text{reserved}(t) \;-\; \max_t \text{allocated}(t).
$$

Because the trace starts with zero allocations, $\max_t \text{reserved}(t)$ is simply the sum of all sizes that appear in `alloc` operations.  
The challenge is to compute the maximum concurrent allocation while processing a sequence of interleaved `alloc` and `free` events.

## Task

Implement `fragmentation_gap(trace)`:

```python
def fragmentation_gap(trace: list[tuple[str, int]]) -> int:
    ...
```

* `trace` is a list of operations.  
  Each operation is a tuple `(op, value)`.  
  * If `op == "alloc"`, then `value` is the size in bytes to allocate.  
  * If `op == "free"`, then `value` is the allocation id returned by a previous `alloc`.  
    Allocation ids start at `0` and increase by one for each successful allocation.

The function must return an integer equal to the fragmentation gap defined above.

## Example

```python
trace = [
    ("alloc", 10),   # id 0
    ("alloc", 20),   # id 1
    ("free", 0),
    ("free", 1)
]
gap = fragmentation_gap(trace)   # -> 0
```

Explanation:  
`reserved = 30`, the maximum concurrent allocation is also `30`, so the gap is `0`.

```python
trace = [
    ("alloc", 5),    # id 0
    ("alloc", 15),   # id 1
    ("free", 0),
    ("alloc", 10),   # id 2
    ("free", 1),
    ("free", 2)
]
gap = fragmentation_gap(trace)   # -> 5
```

Explanation:  
`reserved = 30`, the peak concurrent allocation occurs after the third operation and is `25`.  
Thus `gap = 30 - 25 = 5`.

## What the gate checks

The grader computes a reference result by replaying the trace with an exact algorithm.  
Your implementation must return exactly that integer for all provided test cases.
