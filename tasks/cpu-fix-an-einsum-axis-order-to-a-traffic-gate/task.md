## Context

An einsum-style contraction like $Y_{bi} = \sum_j X_{bj} W_{ij}$ has three
axes — $b$ (batch), $i$ (output), $j$ (the contracted / summed-over
axis) — and the *math* doesn't care what order you nest the loops over
them in: every $(b, i, j)$ triple contributes to $Y_{bi}$ exactly once
either way, so the sum comes out identical no matter which axis is
outermost. **Memory traffic cares enormously.**

$X$ is stored row-major as $B \times J$, so $X_{bj}$ is contiguous along
$j$. $W$ is stored row-major as $I \times J$, so $W_{ij}$ is also
contiguous along $j$. If $j$ is the *innermost* loop, both operands are
scanned with unit stride, and — as a bonus — the whole row $X_{b,:}$
stays cache-resident across all $I$ reuses within a fixed $b$ (nothing
else evicts a 2KB row from an 8KB cache). Move $j$ to the *outermost*
position instead, with $b$ innermost, and $X_{bj}$'s address now jumps
by a full row ($J \times 4$ bytes) on every single step — the same
total set of addresses, visited in an order that turns two contiguous
scans into two strided ones.

## Task

`solve.cpp` contains a buggy `einsum_bij` (declared in `sol.hpp`): the
axis order is wrong. Its loop nest is `j` outer, `i` middle, `b` inner
— the contraction axis outermost, the batch axis innermost. The result
it computes is numerically correct (every triple is still visited
exactly once), but the traffic is not. **Fix the axis order**: reorder
the three loop headers so `b` is outermost, `i` is the middle loop, and
`j` (the contraction axis) is innermost. Leave the loop *body*
unchanged — same three `touch()` calls, same accumulation into `Y`.

## Example

For a single fixed $b$: with $j$ innermost, the inner loop scans
$X_{b,0}, X_{b,1}, \dots, X_{b,J-1}$ in address order — one row, 2048
bytes, 32 cache lines, loaded once and reused for all $I$ values of $i$.
With $j$ outermost and $b$ innermost (the bug), by the time control
returns to process the next $j$ for this $b$, dozens of *other*
$b$ values' addresses have been touched in between — nothing about
$X_{b,j}$ is still resident, so every access is a fresh fetch.

## What the gate checks

The driver multiplies a $32 \times 512$ `X` against a $32 \times 512$
`W` (65536 bytes each, 16x the 8192-byte cache), computes
`checksum = sum(Y)`, and prints it alongside the miss count from a
fresh 8192-byte cache. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both the printed checksum and miss count match the reference}
$$

Both orderings produce the exact same `checksum = 6.780485` (the
contraction's result never depended on loop order) — the gate lives
entirely in the miss count. The buggy `j,i,b` order measures **557179**
misses; the fixed `b,i,j` order measures **33856** — over 16x fewer,
from touching the exact same addresses in a different sequence.
