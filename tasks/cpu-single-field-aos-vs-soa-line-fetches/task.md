## Context

Reading a single field out of every record in an Array-of-Structs (AoS)
array still pulls a WHOLE cache line per record: the field you want sits
next to fields you don't, and the cache doesn't know to fetch less than a
full line. If a record is as wide as (or wider than) one cache line, every
single record costs its own line fetch — even though the data you
actually needed is a fraction of that line.

Struct-of-Arrays (SoA) fixes this for a single-field access pattern by
storing just that field, for every record, packed contiguously. Now
several records' worth of the field share one line, and the cache line
you already paid for keeps paying off for several iterations instead of
one.

## Task

Implement

```cpp
void soa_field_touch(int N, long soa_base);
```

which touches (via `touch()`, declared in `sol.hpp`) a fresh, contiguous
array of `N` 4-byte elements starting at `soa_base` — element `i` at
`soa_base + i*4` — for every `i` in `[0, N)`, **exactly once each**, in
increasing order.

## Example

For `N=2000, soa_base=128000`: element `100` lives at
`128000 + 100*4 = 128400`. With a 64-byte cache line, elements `100`
through `115` all share the same line — touching them in order costs one
miss for the whole group of 16, not 16 misses.

## What the gate checks

`main.cpp` reads one 4-byte field (at a fixed offset) out of 2000
records of a 64-byte-per-record AoS array (a realistic wide record, e.g.
an embedding row — one full cache line per record) with a fixed harness
baseline, and runs your `soa_field_touch` over the same 2000 logical
values laid out contiguously instead — each against its own fresh
2048-byte (64-byte line, 8-set, 4-way) cache — and prints both miss
counts. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. On this fixture the AoS
baseline measures `aos_misses=2000` (one miss per record — the whole
point of AoS is that a full line always has to be fetched for that one
field) while a correct SoA pass measures `soa_misses=125` — a 16x
reduction, purely from letting 16 elements per line survive a stride-1
scan instead of a stride-64 one.
