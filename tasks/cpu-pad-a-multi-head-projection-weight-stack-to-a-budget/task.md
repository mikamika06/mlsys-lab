## Context

A multi-head projection weight stack packs `H` heads' data back-to-back:
head `h`'s region starts at `h * stride` bytes. In a real transformer,
`stride` is naturally a **power of two** (head dims are chosen to be
power-of-two for other reasons entirely). A set-associative cache maps a
byte address to a set via `(address / line_bytes) mod sets`, and `sets`
is *also* a power of two — so when `stride` is a multiple of
`line_bytes * sets`, EVERY head's data lands in exactly the SAME cache
set. Gather the same row across all `H` heads twice in a row (compute
something, then use it right after) and you get thrashing: the cache's
associativity (`ways`) can only hold a few heads' lines at once, so heads
beyond that get evicted before the second pass ever reaches them — even
though the entire per-row working set (`H` floats) is minuscule.

Adding a few bytes of **padding** after each head's region breaks the
power-of-two alignment, spreading the `H` heads' addresses across more
cache sets so they stop competing with each other for the same `ways`
slots.

## Task

Implement

```cpp
int choose_padding_bytes(int H, int row_bytes, int line_bytes, int sets,
                          int ways, int max_pad_bytes);
```

Search every `pad` in `{0, 4, 8, ..., max_pad_bytes}` (step 4 bytes,
inclusive). For each candidate, measure — using the REAL cache model via
`touch()` / `reset_cache()` / `miss_count()` (declared in `sol.hpp`),
never a hand-rolled prediction — how many misses this exact pattern
produces against a fresh `(line_bytes, sets, ways)` cache:

```cpp
reset_cache(line_bytes, sets, ways);
for (h = 0; h < H; h++) touch(h * (row_bytes + pad));       // pass 1
for (h = 0; h < H; h++) touch(h * (row_bytes + pad));       // pass 2, same addresses
```

Return the `pad` with the FEWEST misses; break ties by returning the
SMALLEST such `pad`.

## Example

`H=16, row_bytes=256, line_bytes=64, sets=8, ways=4`: at `pad=0`,
`stride=256=4 lines`, and `(h*4) mod 8` only ever produces `0` or `4` —
all 16 heads alias into just 2 of the 8 sets, 8 heads competing for 4
ways, so pass 2 misses on more than half of them. A well-chosen `pad`
spreads the 16 heads across more sets so pass 2 hits almost everything
instead.

## What the gate checks

`main.cpp` calls `choose_padding_bytes` on two fixed scenarios and then
runs the REAL kernel — the same two-pass gather, for every row of a
64-row (scenario 1) or 32-row (scenario 2) stack, on one continuous cache
session — using the returned `pad`, printing both the chosen `pad` and
the resulting total miss count. The candidate's full stdout is compared
byte-for-byte (`exact_match = 1.0`) against the reference's. On scenario
1 (`H=16, row_bytes=256`, a 2048-byte cache) the reference's search finds
`pad=8`, bringing the total from **2048 misses at `pad=0`** (every single
touch misses, 0% hit rate) down to **171** — better than 10x fewer, from
choosing a padding value 32x smaller than the row itself. Returning any
fixed padding without actually searching, or searching but ranking
candidates by a hand-derived guess instead of the real measured miss
count, lands on a different (or no) minimum and prints the wrong `pad`,
the wrong `total_misses`, or both.
