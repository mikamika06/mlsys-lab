## Context

A cache does not track individual bytes; it tracks fixed-size **lines**
(64 bytes on most x86 parts, 128 on Apple Silicon). Two accesses that
land anywhere within the same line — even to different bytes — hit the
same resident block. The number of *distinct* lines an access trace
touches is the minimum possible number of misses that trace could ever
produce (a perfectly-warmed, infinitely-associative cache would miss
exactly once per distinct line, on first touch).

## Task

Implement

```cpp
long count_distinct_lines(const long* addrs, int n, int line_bytes);
```

Two byte addresses fall in the same line iff `addr / line_bytes`
(integer division) is equal. Return the number of distinct values of
`addrs[i] / line_bytes` over `i in [0, n)`.

## Example

With `line_bytes = 64`, addresses `0, 4, 8` all divide to line `0`
(the same line); `63` also divides to line `0` while `64` divides to
line `1` — the boundary between lines 0 and 1 falls between bytes 63 and
64, not at a "nice" address. A repeated address contributes no new line.

## What the gate checks

`exact_match`: the driver prints the distinct-line count for a fixed
8-address trace (with same-line repeats, a straddled boundary, and
gaps). Using `addr` directly instead of `addr / line_bytes`, or counting
duplicate addresses as separate lines, gives the wrong count; a starter
returning `0` fails outright since the reference finds 4 distinct lines.
