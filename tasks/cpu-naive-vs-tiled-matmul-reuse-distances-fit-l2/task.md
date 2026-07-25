## Context

**Reuse distance** (a.k.a. stack distance) of a memory access is the
number of *distinct* cache lines touched since the last time that same
line was touched. It's the classic tool for predicting cache behavior
without hardware: under an idealized fully-associative LRU cache with
capacity $C$ lines, an access is **guaranteed to hit** iff its reuse
distance is less than $C$ — nothing could have evicted it, because
fewer than $C$ *other* lines were touched in between.

Consider $Y_{ip} = \sum_j A_{ij} V_{pj}$ — an $N \times N$ matrix $A$
matched against a handful ($P$, small and fixed) of query rows $V$.
Looped **naively** ($p$ outermost, $i$ middle, $j$ inner), $A$ doesn't
depend on $p$ at all — so every one of the $P$ passes re-sweeps *all of*
$A$ from the top. The reuse distance for any element of $A$ between one
pass and the next is (up to) the size of the **entire matrix** —
$O(N^2)$ — which blows straight past any fixed cache budget once $N$ is
large enough, no matter how small $P$ is.

**Tiling** $A$ into fixed $T \times T$ blocks and running all $P$ passes
back-to-back over *one block* before moving to the next changes this
completely: once a block is finished, the computation never returns to
it, so every element of $A$'s reuse story is confined to its own
block's visit — bounded by roughly $T^2$, **independent of $N$**.

## Task

Implement, declared in `sol.hpp`:

```cpp
long max_reuse_distance(const long* addrs, int n);
```

For each access `i` in the trace, find the closest earlier index `j` (
`j < i`) whose address lies in the *same* 64-byte line
(`addrs[j] / 64 == addrs[i] / 64`). Its reuse distance is the number of
**distinct** 64-byte lines touched by any index `k` with `j < k < i`
(not counting the line itself). An access with no earlier same-line
touch contributes nothing. Return the **maximum** reuse distance over
the whole trace (`0` if no line is ever touched twice).

## Example

Trace of lines (already divided by 64): `[A, B, A, C, D, A]`. Index 2's
previous same-line touch is index 0; between them (index 1) exactly one
distinct line (`B`) was touched — reuse distance `1`. Index 5's
previous same-line touch is index 2; between them (indices 3, 4) two
distinct lines (`C`, `D`) were touched — reuse distance `2`. The
maximum over this trace is `2`.

## What the gate checks

The driver builds, for `N` in `{16, 32, 64, 128, 256}`, both the naive
and the tiled (8x8 blocks, `P=4`) access traces for `A` described above,
calls `max_reuse_distance` on each, and checks it against this track's
standard 128-line (8192-byte, 64-byte-line) L2 budget. It prints the
distance and fits/doesn't-fit verdict for both, at every size. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed distance and verdict matches the reference}
$$

On this fixture, naive's max reuse distance is `15, 63, 255, 1023,
4095` for the five sizes — fitting the 128-line budget only up to
`N=32`. Tiled's is `15, 31, 63, 127, 255` — still fitting all the way
up to `N=128`, four sizes deep instead of two, because blocking bounds
the reuse distance by the tile, not by how big the whole matrix is.
