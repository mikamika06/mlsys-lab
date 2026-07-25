## Context

The *working set* of a program over some window of time is the set of
distinct memory it touches during that window. Denning's working-set
model uses this directly: to run a window of $W$ consecutive accesses
without a single capacity miss, the cache needs to hold every distinct
line that window touches -- no more, no less. Two addresses fall in the
same cache line iff `addr / line_bytes` is equal.

$$
\text{WS}(t, W) = \bigl|\{\, \lfloor a_i / L \rfloor : t \le i < t+W \,\}\bigr| \cdot L
$$

where $a_i$ is the $i$-th byte address, $L$ is `line_bytes`, and $t$
ranges over every window start. The quantity that matters for sizing a
cache is the **peak**: $\max_t \text{WS}(t, W)$, the biggest working set
the loop this trace came from *ever* needs at once.

## Task

Implement

```cpp
long max_working_set_bytes(const long* addrs, int n, int line_bytes, int W);
```

For every window of `W` consecutive accesses in `addrs[0..n)`
(`t` from `0` to `n-W` inclusive), count the distinct cache lines
(`addrs[i] / line_bytes`) it contains, multiply by `line_bytes`, and
return the maximum of that quantity over all windows.

## Example

With `line_bytes=64`, `W=4` and the trace
`[0, 64, 0, 64, 0, 64, 0, 64, 128, 192, 256, 320, 384, 448, 512, 576]`:
the first 8 accesses bounce between line 0 and line 1 (any window there
has at most 2 distinct lines -> 128 bytes), but the last 8 accesses
stream through 8 brand-new lines with no repeats, so every window fully
inside that region has 4 distinct lines -> 256 bytes. The maximum over
all windows is 256, not 128.

## What the gate checks

`main.cpp` runs the trace above with `line_bytes=64`, `W=4`, and prints
`max_working_set_bytes`. The reference gets 256. Two shortcuts both
fail: counting distinct lines across the *whole* trace instead of the
best single window gives 640 (10 distinct lines, ignoring that a window
can only see `W` accesses at a time); checking only the first window
(or any one fixed window) gives 128, missing the bigger working set
later in the trace. A starter returning `0` fails outright.
