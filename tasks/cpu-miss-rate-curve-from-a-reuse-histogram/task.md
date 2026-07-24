## Context

A **reuse-distance histogram** summarizes a memory access trace without
committing to any particular cache size: for every access, count the
number of *distinct* other addresses referenced since that same address
was last touched (its reuse distance), and bucket the access into
`hist[reuse_distance]`. Accesses that touch an address for the very first
time have no previous occurrence — they go into a separate `cold_misses`
count, since they are compulsory misses at *every* cache size.

The payoff: this single histogram lets you compute the miss rate at *any*
fully-associative LRU cache size, without re-simulating the trace. Under
an LRU cache of `cache_size` lines, an access with reuse distance $d$ is a
**hit** iff $d < \text{cache\_size}$ (its address is still within the top
`cache_size` slots of the LRU stack) and a **miss** otherwise. So the
**miss-rate curve** (MRC) at a given cache size is a *tail sum* of the
histogram:

$$\text{misses}(C) = \text{cold\_misses} + \sum_{d = C}^{\infty} \text{hist}[d] \qquad \text{miss\_rate}(C) = \frac{\text{misses}(C)}{\text{total\_accesses}}$$

As $C$ grows, the tail sum only shrinks (never grows) — the MRC is
monotonically non-increasing, with a "knee" wherever the histogram has a
mass of accesses at that exact reuse distance.

## Task

Implement

```cpp
double miss_rate_at_cache_size(const long* hist, int max_dist, long cold_misses,
                                long total_accesses, int cache_size);
```

Return `(cold_misses + sum of hist[d] for d in [cache_size, max_dist))) /
total_accesses`.

## Example

A trace that round-robins through 8 distinct blocks, repeated many times,
produces every repeat access at reuse distance exactly 7 (7 *other*
distinct blocks are touched between two visits to the same block). So the
whole histogram mass sits at `hist[7]`, plus 8 cold misses (one per block,
first cycle). At `cache_size = 7`: every repeat access has `d = 7 >= 7`, a
miss — so the miss rate is `1.0`. At `cache_size = 8`: now `7 < 8`, every
repeat is a hit — the miss rate drops to just `cold_misses / total`.

## What the gate checks

`max_abs_err <= 1e-9` on 5 fixed cache-size queries (`1, 4, 7, 8, 16`)
against the histogram of a real 160-access, 8-block round-robin trace
(built by the driver with a plain, obviously-correct O(n^2) scan). The
curve is flat at `1.0` for `cache_size <= 7`, drops sharply to `0.05` at
`cache_size = 8`, and stays flat there for `cache_size = 16` — summing
from the wrong end (`d < cache_size` instead of `d >= cache_size`),
off-by-one on the boundary, or forgetting `cold_misses`, all move at least
one of the 5 printed rates well past the tolerance.
