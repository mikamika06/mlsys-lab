## Context

During autoregressive decode, each new token's attention step re-reads
*every* past token's key/value entry from the KV cache — a full,
sequential re-scan that grows by one record every step. A software
prefetcher can hide this: while demanding record `t`, also issue a
prefetch for record `t + D` (the "prefetch distance") so it's already
resident by the time the scan reaches it.

`D` has a real Goldilocks zone. Too small, and there's barely any
lookahead — reads still mostly hit cold cache. Too large, and the
prefetch fires so far ahead of the corresponding demand read that,
by the time the scan actually gets there, the finite cache has long since
evicted it to make room for everything prefetched (and demanded) in
between — the prefetch was real work for zero benefit.

## Task

Implement both, using the shared `touch()` / `cache_reset()` cache model
declared in `sol.hpp`:

```cpp
long simulate_decode_pass(int T, int rec_bytes, int prefetch_distance);
int  choose_best_prefetch_distance(int T, int rec_bytes, int max_distance);
```

`simulate_decode_pass` resets the cache, then scans records `0..T-1` in
order: demand-touch record `t` (counting a miss if it wasn't resident),
then, if `prefetch_distance > 0` and in bounds, touch record
`t + prefetch_distance` too (never counted). `choose_best_prefetch_distance`
tries every distance from `0` to `max_distance` and returns whichever
minimizes the miss total from `simulate_decode_pass` (smallest distance
on a tie).

## Example

With 64-byte cache lines and 32-byte records, two consecutive records
already share a line, so even `D=0` (no prefetching) isn't a total loss —
but it still misses on every *other* record once the working set outgrows
the cache. Prefetching just 1-3 records ahead keeps the scan almost
entirely resident (misses drop from double digits to nearly zero);
prefetching 17+ records ahead — more than the whole cache can hold at
once — is no better than not prefetching at all, since every prefetched
record is evicted long before its demand read arrives.

## What the gate checks

`exact_match` on `(naive_misses, best_d, best_misses)` for a fixed 32-record
scan over a 64-byte-line, 4-set, 2-way cache (16-record capacity).
Reference: `naive_misses=16`, `best_d=1`, `best_misses=1`. Prefetching the
wrong record (e.g. `t - D` or `t` itself), counting prefetch touches as
misses, or searching the wrong direction (largest-miss instead of
smallest), all change at least one of the three printed numbers.
