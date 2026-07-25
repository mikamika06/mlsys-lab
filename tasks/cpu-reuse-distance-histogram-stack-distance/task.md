## Context

Temporal locality — "will this line be touched again soon?" — has a
precise, cache-size-independent measure: **stack distance** (also called
LRU reuse distance). The stack distance of an access is the number of
*distinct* other lines touched since that same line was last referenced.
An access with stack distance $d$ would hit in any fully-associative LRU
cache with capacity $\ge d+1$ lines, and miss in any smaller one — so a
histogram of stack distances across a whole trace tells you, for every
possible cache size at once, exactly what fraction of accesses would hit.

An access to a line that has never appeared before has no previous
reference at all — it's **cold**, with no finite stack distance (it
misses in a cache of any size).

## Task

Implement:

```cpp
void stack_distance_histogram(const long* addrs, int n, int line_bytes,
                               int num_lines, long* hist_out);
```

For each access `i`, let `line_i = addrs[i] / line_bytes`. Its stack
distance is the count of distinct lines touched strictly between the
PREVIOUS access to `line_i` (exclusive) and access `i` (exclusive) — an
immediate repeat (nothing else touched in between) has stack distance
$0$. The trace only ever touches `num_lines` distinct lines, so no stack
distance can exceed `num_lines - 1`. Fill:

- `hist_out[0]` = number of cold accesses
- `hist_out[1 + d]` = number of accesses with stack distance exactly $d$,
  for $d \in [0, \text{num\_lines} - 1]$

## Example

The driver (`main.cpp`, fixed) runs the 12-access trace over 6 lines:

$$0, 1, 2, 1, 3, 0, 2, 4, 4, 5, 0, 1$$

Access `1` (line 2, cold), `3` (line 1, last seen at index 1 — only line 2
appeared in between $\Rightarrow$ distance 1), `7`→`8` (line 4 twice in a
row $\Rightarrow$ distance 0), and the final access (line 1, last seen at
index 3 — lines 3, 0, 2, 4, 5 all appeared in between $\Rightarrow$
distance 5) are representative:

```
cold=6
dist0=1
dist1=1
dist2=0
dist3=3
dist4=0
dist5=1
```

$6 + 1 + 1 + 0 + 3 + 0 + 1 = 12$ — every access lands in exactly one bin.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed bin count to `exact_match` the same
driver linked against the reference implementation. Miscounting which
references are "cold" versus distance-0 (or including the referenced line
itself, or the endpoint accesses, in the distinct-lines count) shifts
counts between bins and fails the gate immediately, even though the total
across all bins still sums to 12. The starter returns an all-zero
histogram.
