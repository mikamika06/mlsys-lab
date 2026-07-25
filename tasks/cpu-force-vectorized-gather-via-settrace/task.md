## Context

A **gather** — $\text{output}[i] = \text{table}[\text{indices}[i]]$ for every
$i$ — is cheap in principle, but a naive scalar loop pays for a fresh memory
access on *every* output position, even when `indices` repeats the same
value many times (a very common pattern: a hot vocabulary id, a popular
row, a token that recurs across a batch). A real vectorized/SIMD gather
instruction, or a hand-written lookup that keeps recently-fetched values
around, doesn't pay that cost twice — once it has fetched `table[v]` for
some value `v`, every later repeat of `v` is served from a register/cache
it already holds, with no additional memory access at all.

## Task

Implement

```cpp
void gather_dedup(const float* table, int table_len, const int* indices,
                   int n, float* output);
```

`output[i]` must equal `table[indices[i]]` for every `i` in `[0, n)` —
repeats included. But you must call `touch(table_addr(indices[i]))`
(declared in `sol.hpp`) **only the first time a given index VALUE is
looked up** anywhere during the call. Keep your own bookkeeping — e.g.
two arrays of size `table_len`, one marking "have I fetched this index
before" and one holding "what value did I fetch for it" — and serve every
later repeat of that value from your own arrays instead of touching
memory again.

## Example

`indices = [5, 9, 5, 5, 9]`: the first `5` and the first `9` each cost one
`touch()`; the remaining three accesses (`5`, `5`, `9`) must come from
your own cached values with **no** further `touch()` calls, yet
`output = [table[5], table[9], table[5], table[5], table[9]]` exactly as
a plain gather would produce.

## What the gate checks

`main.cpp` builds a 4096-float table (16384 bytes, 8x the modeled cache)
and 4000 indices that cycle through only 64 DISTINCT values
(`HOT[j] = j*61 mod 4096`, a coprime stride so all 64 are genuinely
distinct and spread across the table; `indices[i] = HOT[i % 64]`) — a hot
working set reused across a long batch. It gathers with a fixed
always-touch harness baseline and with your `gather_dedup`, each against
its own fresh 2048-byte (64-byte line, 8-set, 4-way LRU) cache, and
prints both touch counts, both miss counts, and a position-weighted
checksum of your output (sensitive to a value landing at the wrong
index). The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's.

On this fixture the naive always-touch gather measures
`naive_touches=4000` and, because the 64-line working set thrashes
against a cache that can only hold 32 lines at once, `naive_misses`
close to 4000 too — nearly every repeat re-misses. A correct
`gather_dedup` measures exactly `dedup_touches=64` (one per distinct
index value, ever) and `dedup_misses=64` (each of those 64 is a
compulsory first-touch miss, but none of them is ever paid twice) — a
~60x reduction in touches from recognizing repeats instead of
re-fetching them.
