## Context

A cache maps an address to a set using its low-order bits (after the line
offset): `set = (address / line_bytes) % num_sets`. When a matrix is stored
row-major with leading dimension (row stride) `ld`, moving from row `i` to
row `i+1` at a fixed column jumps the address by `ld` elements. If
`ld * elem_bytes` happens to be a multiple of `num_sets * line_bytes`, that
jump lands in the exact same set every time — **every row aliases onto one
set**, no matter how much *total* cache capacity is sitting idle elsewhere.
With only a handful of ways per set, a working set that would easily fit the
whole cache instead thrashes a single set to death.

The classic real-world trigger is picking `ld` equal to a "round" size — a
power of two, or a multiple of the matrix width — because that is the
obvious, natural choice when nothing forces you to think about the cache.
The classic fix is **padding**: make `ld` slightly larger than the natural
width so consecutive rows land in *different* sets.

## Task

`solve.cpp` computes the sum of an `R x C` matrix, sweeping it **column
by column** (outer loop over column `j`, inner loop over row `i`) — this
traversal order revisits the same handful of cache lines repeatedly across
nearby columns, so it rewards a cache-friendly layout and punishes a
thrashing one. It picks its own leading dimension `ld` for the matrix it
allocates, but currently just uses `ld = C`:

```cpp
double sum_all_columns(int R, int C) {
    int ld = C;  // <-- picks LD with no regard for the cache
    ...
}
```

With the driver's fixed cache geometry (32 sets, 4 ways, 64-byte lines = 8
doubles/line) and `C = 256` (a multiple of `32 sets * 8 doubles/line = 256`),
this aliases *every* row of *every* 8-column block onto the same 4-way set,
and the sweep thrashes on essentially every access.

Fix `sum_all_columns` so it pads `ld` whenever the natural width is a
multiple of the dangerous stride (`32 * 8 = 256`), bumping it by one cache
line's worth of doubles (`+8`) to break the alignment — while still
filling and summing every element with
`value(i, j) = (i * 131 + j * 977) % 1009` (a formula that depends only on
`i` and `j`, never on `ld`, so the sum is identical no matter what leading
dimension you pick).

## Example

With `R = 64`, `C = 256`, the fixed driver reports:

```
R=64 C=256
sum=8254690.0
misses=2048
```

`misses=2048` is exactly `R*C/8`: with `ld` padded to 264, each 64-byte
line holds 8 consecutive columns of one row, so the fixed column-major sweep
touches every line once as a cold miss and hits on the other 7 accesses to
it. The buggy `ld = 256` version prints the same `sum=8254690.0` (the totals
never depend on `ld`) but `misses=16384` — with every row of every 8-column
block aliased onto one 4-way set, essentially every single access misses.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output — including `misses=`, not
just `sum=` — to match the reference (`main.cpp` + `ref.cpp`) byte-for-byte
(`exact_match == 1.0`). A fix that gets the sum right but leaves `ld = C`
unchanged (or pads by an amount that is itself a multiple of 256, which
re-creates the same alias) still prints `misses=16384` and fails the gate —
the padding has to actually change which set each row lands in.
