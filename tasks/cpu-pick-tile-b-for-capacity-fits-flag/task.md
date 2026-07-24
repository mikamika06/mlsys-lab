## Context

The inner loop of a blocked matmul (or any 3-operand blocked kernel) works
on three $B \times B$ tiles at once — a tile of $A$, a tile of the other
operand, and a tile of the output — and gets its speedup only if all three
stay resident in cache for the whole inner loop. Pick $B$ too small and
you leave cache capacity (and reuse) on the table; pick it even one step
too large and the tiles no longer fit, eviction starts, and the "blocked"
kernel is no better than the naive one.

For three $B \times B$ tiles of `elem_size`-byte elements to fit in
`capacity_bytes` of cache:

$$3 \cdot B^2 \cdot \text{elem\_size} \le \text{capacity\_bytes}$$

The largest integer $B$ satisfying this is

$$B = \left\lfloor \sqrt{\dfrac{\text{capacity\_bytes}}{3 \cdot \text{elem\_size}}} \right\rfloor$$

## Task

`sol.hpp` gives you a deterministic fully-associative LRU cache model
(`reset_cache()` / `touch_byte(addr)` / `miss_count()`: `capacity_bytes /
64` lines of 64 bytes each). Implement:

```cpp
int derive_tile_b(long capacity_bytes, int elem_size);
```

Return the largest integer $B$ such that $3 B^2 \cdot \text{elem\_size}
\le \text{capacity\_bytes}$.

The driver (`main.cpp`, fixed) calls your `derive_tile_b` with
`capacity_bytes = 4096`, `elem_size = sizeof(float) = 4`, allocates three
independently 64-byte-aligned `B * B`-float tiles, touches every element
of all three tiles once (pass 1, necessarily cold), then touches the exact
same elements again in the exact same order (pass 2) — and checks whether
pass 2 added any NEW misses. If the three tiles genuinely fit in the
4096-byte cache, nothing gets evicted between passes and pass 2 adds zero
misses (`fits = 1`); if $B$ is even one too large, the tiles overflow the
cache, evict each other, and pass 2 re-misses almost everything (`fits =
0`).

## Example

$B = \left\lfloor \sqrt{4096 / 12} \right\rfloor = \lfloor 18.475\dots
\rfloor = 18$. Three $18 \times 18$ float tiles are $3 \times 1296 = 3888$
data bytes, but each tile is aligned separately, so each one alone rounds
up to $\lceil 1296 / 64 \rceil = 21$ lines — $63$ lines total, just barely
under the cache's $4096 / 64 = 64$-line capacity:

```
B=18
misses_pass1=63
misses_pass2=63
fits=1
```

One step larger, $B = 19$ ($3 \times 19^2 \times 4 = 4332 > 4096$, so it
should NOT fit) needs $3 \times \lceil 361 \cdot 4 / 64 \rceil = 69$ lines
— over capacity, so pass 2 evicts and re-misses everything:

```
B=19
misses_pass1=69
misses_pass2=138
fits=0
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number — `B`, both miss counts, and
`fits` — to `exact_match` the same driver linked against the reference
derivation. An off-by-one $B$ (forgetting the floor, or rounding up) still
compiles and runs, but `fits` flips to `0` and the miss counts diverge, so
the gate catches it even without knowing the "right" formula in advance.
The starter returns `0`, so all three tiles are empty and every number
comes out wrong.
