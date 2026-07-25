## Context

A **cache-oblivious** algorithm never mentions a cache size, line size, or
associativity — it recurses, splitting the problem into geometrically
smaller pieces until the pieces are small enough that whatever cache is
actually running the program holds one comfortably. For transposing an
$N \times N$ matrix, $\mathrm{out}[c][r] = \mathrm{in}[r][c]$, that means
recursively splitting the $n \times n$ index space into four
$(n/2) \times (n/2)$ quadrants and recursing on each, only falling back to
a direct double loop once a quadrant is small enough.

Getting the *values* right is necessary but not sufficient here — a flat,
unblocked double loop computes the exact same transpose, correctly, while
touching memory in a much worse order. The real test of a cache-oblivious
implementation is whether its *access pattern* achieves the same locality
as a hand-tuned blocked version, without ever being told the block size.

## Task

Implement

```cpp
void co_transpose(const float* in, float* out, int N);
```

which computes $\mathrm{out}[c][r] = \mathrm{in}[r][c]$ for the full
$N \times N$ matrix ($N$ a power of two, $N > 8$) by recursively splitting
the *global* index space $[r_0, r_0+n) \times [c_0, c_0+n)$ into four
$(n/2) \times (n/2)$ quadrants once $n > 8$, and recursing on each (any
order). When $n \le 8$ (the base case), for every `(row, col)` pair in the
block:

1. actually copy the data: `out[col*N + row] = in[row*N + col];`
2. `touch(in_addr(N,row,col))` then `touch(out_addr(N,col,row))` (both
   declared in `sol.hpp`) — **exactly once each**, matching step 1.

`N` never changes across the recursion — only the quadrant's own
`(r0, c0, n)` does — because addressing always needs the *full* matrix's
row stride.

## Example

For $N = 64$: the top call splits into four $32\times32$ quadrants, each
into four $16\times16$, each into four $8\times8$ — which hit the base
case and get copied + touched directly. That's $4^3 = 64$ leaf blocks of
$8\times8=64$ pairs each: $64\times64\times2 = 8192$ total touches (one
read + one write per element), exactly matching a flat double loop in
*what* gets touched — only the *order*, and therefore the miss count,
differs.

## What the gate checks

`main.cpp` fills a real 64x64 matrix with exact, float-representable
integer values, transposes it with a fixed 8x8-**blocked** harness
baseline and with your `co_transpose` — each against its own fresh
8192-byte (64-byte line, 32-set, 4-way) cache model — and prints both
miss counts plus a position-weighted checksum of your output (sensitive
to a wrong *permutation*, since a plain sum of the values wouldn't
change even if elements landed in the wrong slots). The candidate's full
stdout is compared byte-for-byte (`exact_match = 1.0`) against the
reference's. On this fixture the reference measures `blocked_misses=768`
and `co_misses=512` — the cache-oblivious recursion reaches the same
*optimal* cold-miss floor ($32768$ bytes $/$ $64$-byte line $= 512$) as
the blocked version, **without ever being told the block size**. A
flat, unblocked double loop computes the exact same (correct!) checksum
but measures `co_misses=4352` — over 8x worse — because getting the
*values* right doesn't require getting the *access order* right.
