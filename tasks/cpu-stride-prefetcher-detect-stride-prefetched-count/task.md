## Context

A real stride prefetcher does not track one stream — it keeps a small
table of entries (indexed by, e.g., the load's PC), each independently
learning the address delta of its own stream. Streams get interleaved
in the actual instruction trace, so the table has to keep each stream's
last address and confirmed delta separate, or one stream's accesses will
corrupt another's state and either miss real patterns or "confirm" a
fake one.

## Task

Implement

```cpp
long stride_prefetch_count(const int* stream_id, const long* addr, int n, int num_streams);
```

For each access `i` (`stream_id[i]`, `addr[i]`), per `sol.hpp`: the
first access of a stream just records its address; the second records
the delta from the first; every access after that compares its delta to
the stream's recorded delta — a match issues (and counts) a prefetch,
and either way the recorded delta updates to the current one. Return the
total prefetch count across all streams.

## Example

Stream 0 (constant stride 64 over 5 accesses) confirms its pattern on
the 3rd access and prefetches on the 3rd, 4th, and 5th (3 prefetches).
Stream 1 (constant stride 64 over 3 accesses) confirms on its 3rd access
(1 prefetch). Stream 2's deltas are `100` then `200` — never repeat, so
it never confirms a pattern (0 prefetches), even though it shares the
trace with two streams that do.

## What the gate checks

`exact_match`: the driver prints the total prefetch count for one fixed
11-access, 3-stream interleaved trace. Sharing state across streams (or
tracking only one stream), comparing against the wrong delta, or
prefetching on the 2nd access instead of the 3rd all change the count;
the reference finds `4`, and a starter returning `0` fails outright.
