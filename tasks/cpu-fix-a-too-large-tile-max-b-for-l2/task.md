## Context

A blocked (tiled) matmul's inner loop only pays off if the tiles it
reuses actually stay in cache between reuses. For a classic
`C[i][j] += A[i][k] * B[k][j]` blocking, the inner loop over a `B x B`
tile keeps **three** `B x B` tiles of 4-byte floats resident at once — a
tile of `A`, a tile of `B`, and the `C` accumulator tile — not just one.
Sizing the tile from `l2_bytes / (bytes per float)` alone, forgetting the
factor of three, produces a `B` that looks plausible but is roughly
`sqrt(3) ~ 1.73x` too large: three tiles of that size blow past L2's
capacity, and the "blocked" kernel spills exactly the way an unblocked
one would, defeating the whole point of tiling.

## Task

`sol.hpp` declares

```cpp
int max_tile_b_for_l2(long l2_bytes);
```

which must return the **largest** integer `B` such that three `B x B`
tiles of 4-byte floats fit together in an L2 of `l2_bytes` bytes:

$$
3 \cdot B^2 \cdot 4 \le \text{l2\_bytes}, \qquad 3 \cdot (B+1)^2 \cdot 4 > \text{l2\_bytes}
$$

The shipped implementation has the classic version of this bug: it
solves `B^2 \cdot 4 \le \text{l2\_bytes}` — sizing as if only *one* tile
needs to fit — so it returns a `B` about `\sqrt{3}` times too large. Fix
it so all three tiles are accounted for.

## Example

For `l2_bytes = 32768`: the correct `B` is `52`
(`3 * 52^2 * 4 = 32448 <= 32768`, `3 * 53^2 * 4 = 33708 > 32768`). The
buggy formula instead solves `B^2 * 4 <= 32768`, giving `B = 90` —
`3 * 90^2 * 4 = 97200`, nearly three times over budget.

## What the gate checks

The driver calls your function with `l2_bytes = 32768` (matching a
modelled 128-set, 4-way, 64-byte-line L2 of exactly that capacity), then
**verifies residency by simulation, not by formula**: it sweeps the
combined byte footprint of the three tiles over a fresh cache twice back
to back — the first (cold-fill) pass, then a second (reuse) pass — for
both the returned `B` and `B + 1`. A tile set is "resident" iff the
second pass adds zero new misses. It prints the returned `B` and both
fit/no-fit flags. The grader compiles `solve.cpp` with `clang++ -O2
-std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{printed } B \text{, fits\_b, and fits\_b\_plus\_1 all match the reference}
$$

The reference prints `b=52 fits_b=1 fits_b_plus_1=0` — `B` fits, `B+1`
doesn't, confirming `52` really is the maximum. The buggy one-tile
formula prints `b=90 fits_b=0 fits_b_plus_1=0` — its own `B` doesn't even
fit, let alone being the maximum that does — which fails the gate on
every field.
