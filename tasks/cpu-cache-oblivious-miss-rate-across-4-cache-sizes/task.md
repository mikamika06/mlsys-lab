## Context

A *cache-aware* algorithm (e.g. a manually tiled loop) is fast only for the
cache size it was tuned for: pick a tile that's too big for a smaller
cache, or too small to exploit a bigger one, and its miss rate suffers. A
*cache-oblivious* algorithm gets around this by never mentioning a cache
size at all: it recurses, splitting the problem into geometrically smaller
pieces, until the pieces are so small that whatever cache is actually
running the program — of whatever line size, capacity or associativity —
holds one comfortably. The same unmodified code is therefore close to
optimal at *every* cache size simultaneously, which is exactly the
property this task measures.

For transposing an $N \times N$ matrix, $\mathrm{out}[j][i] = \mathrm{in}[i][j]$,
the cache-oblivious version recursively splits the $n \times n$ index
space into four $(n/2) \times (n/2)$ quadrants and recurses on each, only
falling back to a direct double loop once a quadrant is small enough (here,
$n \le 8$).

## Task

Implement

```cpp
void co_transpose(int N);
```

which computes $\mathrm{out}[j][i] = \mathrm{in}[i][j]$ for the full
$N \times N$ matrix ($N$ a power of two, $N > 8$) by recursively splitting
the *global* index space $[r_0, r_0+n) \times [c_0, c_0+n)$ into four
$(n/2) \times (n/2)$ quadrants once $n > 8$, and recursing on each. When
$n \le 8$, touch `in_addr(N,row,col)` then `out_addr(N,col,row)` (both
declared in `sol.hpp`) directly for every `(row, col)` pair in the block,
**exactly once each**. `N` never changes across the recursion — only the
quadrant's own `(r0, c0, n)` does — because addressing always needs the
*full* matrix's row stride.

Crucially, `co_transpose` must never branch on a cache parameter: it has
no way to know the driver is about to run it against four completely
different cache capacities.

## Example

For $N = 64$: the top call splits into four $32 \times 32$ quadrants, each
of those into four $16 \times 16$ quadrants, each of those into four
$8 \times 8$ quadrants — which hit the base case and get touched directly.
That's $4^3 = 64$ leaf blocks of $8\times 8 = 64$ pairs each, i.e.
$64 \times 64 \times 2 = 8192$ total touches (one read + one write per
element) — matching a flat double loop exactly in *what* gets touched,
only the *order* differs.

## What the gate checks

The driver builds a fixed 64-byte-line, 4-way cache model and runs your
`co_transpose` on the same 64x64 matrix (32768 bytes of `in` + `out`
combined) against **4 different capacities** — 512B, 2048B, 8192B, 32768B
— resetting and reconfiguring the model between each, and prints the miss
count and miss rate (misses / 8192 total accesses) for all four, forming a
4-point miss-rate curve (MRC):

$$
\mathrm{rate}_k = \frac{\mathrm{misses}_k}{8192}, \quad k = 1..4
$$

On this fixture the reference measures `512→4608 (0.5625)`,
`2048→1680 (0.2051)`, `8192→512 (0.0625)`, `32768→512 (0.0625)` — note the
curve reaches the cold-miss optimum (512 = 32768 bytes / 64-byte line,
one miss per distinct line) already at 8192 bytes, a *quarter* of the
combined working set, and simply **stays** there — no retuning needed to
go from 8192 to 32768. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{all four printed lines match the reference}
$$

A flat loop, a wrong quadrant split, or an incomplete/empty
`co_transpose` (the starter) touches the wrong addresses in the wrong
order and gets a visibly worse — or, for the empty starter, an entirely
absent (all-zero) — miss curve, failing the gate at every one of the 4
sizes at once.
