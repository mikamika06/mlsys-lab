## Context

The *working set* of a program is the collection of distinct memory locations accessed during a period of execution.  
When a loop or algorithm accesses memory in an irregular fashion, the working set size determines how many cache lines must be resident for optimal performance.

For a contiguous sequence of $k$ memory accesses (a *sliding window*) we define its working set size as

$$
W_k = \bigl|\{\, a_i : t-k < i \leq t \,\}\bigr|,
$$

where $a_i$ is the $i$‑th byte address accessed and $t$ indexes the last access in the window.  
The *maximum working set size* over all windows of length $k$ is the largest value of $W_k$ encountered.

## Task

Implement a function that, given an array of byte addresses `addrs` and a positive integer `window_size`, returns the maximum working set size among all contiguous subsequences of length `window_size`.

```python
def max_working_set(addrs: list[int] | tuple[int], window_size: int) -> int:
    ...
```

The function should be efficient for large inputs (e.g., $10^6$ accesses).

## Example

```python
>>> addrs = [1, 2, 3, 2, 4, 5]
>>> max_working_set(addrs, 3)
3
# Windows of length 3:
#   [1,2,3] -> 3 distinct addresses
#   [2,3,2] -> 2 distinct addresses
#   [3,2,4] -> 3 distinct addresses
#   [2,4,5] -> 3 distinct addresses
```

## What the gate checks

The grader computes a reference answer by iterating through all windows of length `window_size` and counting distinct addresses.  
Your implementation must produce an integer that exactly matches this reference. The gate metric is `exact_match`; it returns $1$ if your result equals the reference, otherwise $0$. No other criteria are evaluated.
