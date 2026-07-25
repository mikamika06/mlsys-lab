## Context

A **gather** — `output[i] = table[indices[i]]` for a scattered index array
— has no useful spatial locality if you walk it in the order the indices
happen to arrive in: consecutive `i`'s can point anywhere in `table`, so
consecutive touches can land in completely unrelated cache lines, even
when many of those indices actually share a line with each other. If
`table` doesn't fit in cache, every one of those unrelated touches evicts
something you'll need again before you get back around to it.

**Sorting the indices before gathering** doesn't change *what* gets read
or *where the results go* — `output[i]` still ends up with
`table[indices[i]]` — it only changes the *order* the table is visited
in. Once nearby index values are visited back-to-back, indices that share
a cache line (16 floats per 64-byte line here) get touched while that
line is still resident, turning what would have been several separate
misses into one.

## Task

Implement

```cpp
void gather_sorted(const float* table, int table_len, const int* indices,
                    int n, float* output);
```

Read the table (and call `touch()`, declared in `sol.hpp`) in **any**
order you choose — e.g. sorted by index value — but touch every index in
`indices` **exactly once**, via `touch(table_addr(indices[i]))`, and make
sure `output[i]` ends up holding `table[indices[i]]` for **its own** `i`,
regardless of the order you read the table in.

## Example

`indices = [900, 3, 901, 4]`: read in the ORIGINAL order, this bounces
between two far-apart regions of `table` every single step. Read
**sorted** by value — `3, 4, 900, 901` — the two pairs that share a line
(`3`/`4`, and `900`/`901`) get read back-to-back; `output[0]` still ends
up with `table[900]` and `output[2]` with `table[901]`, exactly as
required — only the *order* of the reads changed, not the result.

## What the gate checks

`main.cpp` builds a 4096-float table (16384 bytes, 8x the cache) and 2000
DISTINCT, scattered indices (`indices[i] = (i*37) mod 4096` — a genuine
permutation, not a repeating pattern), gathers them with a fixed
original-order harness baseline and with your `gather_sorted`, each
against its own fresh 2048-byte (64-byte line, 8-set, 4-way) cache, and
prints both miss counts plus a position-weighted checksum of your output
(sensitive to a value landing at the wrong index — summing the raw values
alone wouldn't catch that, since a gather never invents or drops values).
The candidate's full stdout is compared byte-for-byte (`exact_match =
1.0`) against the reference's. On this fixture the naive original-order
gather measures `naive_misses=2000` — every single touch misses, since
2000 *distinct* indices scattered across a permutation almost never
repeat within a short enough window to reuse anything — while a correctly
sorted gather measures `sorted_misses=256`, matching the true number of
distinct cache lines the 2000 chosen indices touch: roughly 8x fewer
misses, from visiting the exact same table values in a different order.
