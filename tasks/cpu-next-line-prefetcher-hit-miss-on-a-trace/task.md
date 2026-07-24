## Context

A **next-line prefetcher** is the simplest hardware prefetcher there is: on
every access to cache line $L$, it also fetches line $L+1$ speculatively,
betting that the program will keep streaming forward. When that bet is
right, an access that would otherwise have been a compulsory miss turns
into a free hit, because the data is already resident by the time the
demand access for it arrives. When the program jumps somewhere unrelated,
the bet is wasted — worse, it can *evict* something a future access
actually needed, since prefetched data still has to live in the same
finite cache.

## Task

Implement

```cpp
void simulate_next_line_prefetch(const int* line_trace, int n, int cache_lines,
                                  int* hits_out, int* misses_out);
```

exactly per the LRU + prefetch rules in `sol.hpp`: for each access, service
the demand access first (hit/miss, with front-insertion / back-eviction),
then separately issue a next-line prefetch for `line+1` (back-insertion /
back-eviction, no effect on the hit/miss counts).

## Example

With a 3-line cache and trace `0,1,2,3`: accessing `0` misses, but
immediately queues a prefetch for `1`; accessing `1` finds it already
there — a hit — and queues a prefetch for `2`, and so on. A stream that
would miss on *every* access in a plain LRU cache this small instead
misses only once, on the very first line.

## What the gate checks

`exact_match` on `(hits, misses)` from a fixed 9-line, 3-line-cache trace:
a sequential run `0,1,2,3`, a jump to a distant region `50,51,52`, then a
jump back to `4,5`. The sequential portions mostly hit thanks to
prefetching, but the jump back to `4` misses anyway — `4` had been
prefetched earlier, but the intervening jump to `50,51,52` evicted it
under the 3-line capacity before it was ever used. Reference values:
`hits=6 misses=3`. Skipping the prefetch step, prefetching the wrong line
(e.g. `L-1` or `L` itself), or inserting prefetched lines at the MRU end
instead of the LRU end, all change at least one of the two printed counts.
