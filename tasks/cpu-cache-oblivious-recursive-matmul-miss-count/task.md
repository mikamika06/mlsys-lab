## Context

Matrix multiplication $C = A \times B$ done the textbook way — three nested
loops over $i, j, k$ — has a locality problem that gets worse as $N$ grows:
for a fixed $(i, j)$, the inner loop over $k$ reads a whole row of $A$
(contiguous, cheap) but a whole *column* of $B$ (stride $N \times 4$ bytes,
one cache line touched per element). Once $N$ is large enough that a
row/column no longer fits in cache, every $(i, j)$ pair re-walks $B$'s
column from scratch, and almost none of that traffic hits.

**Cache-oblivious** algorithms sidestep this without ever being told the
cache's line size or capacity. `recursive_matmul` splits each $N \times N$
matrix into four $(N/2) \times (N/2)$ quadrants and recurses:

$$
\begin{aligned}
C_{11} &\mathrel{+}= A_{11}B_{11} + A_{12}B_{21} &
C_{12} &\mathrel{+}= A_{11}B_{12} + A_{12}B_{22} \\
C_{21} &\mathrel{+}= A_{21}B_{11} + A_{22}B_{21} &
C_{22} &\mathrel{+}= A_{21}B_{12} + A_{22}B_{22}
\end{aligned}
$$

Each quadrant is a *view* into the original matrix (same underlying
storage, same row stride $N$), not a copy. As the recursion descends, the
working set of each sub-problem keeps shrinking — and at some depth it
drops below the cache's capacity no matter what that capacity actually is.
From that point on, the whole sub-problem's traffic hits cache. That's the
"oblivious" part: the algorithm never queries line size or capacity, it
just gets there by construction.

## Task

Implement two functions declared in `sol.hpp`, both driving accesses
through the harness's `touch_byte(addr)` hook (a deterministic 4-way,
32-set, 64-byte-line LRU cache defined in `main.cpp` — 8192 bytes total):

```cpp
void naive_matmul(int N, long a_base, long b_base, long c_base);
void recursive_matmul(int N, long a_base, long b_base, long c_base);
```

**`naive_matmul`**: the textbook triple loop. For every $i, j, k \in
[0, N)$, in that loop order (i outer, j middle, k inner), call
`touch_byte` on the address of $A[i][k]$, then $B[k][j]$, then $C[i][j]$
— element $(r, c)$ of an $N \times N$ row-major matrix at `base` lives at
`base + (r*N + c)*4`.

**`recursive_matmul`**: cache-oblivious recursion. While $N > 8$, split
each matrix into four $(N/2) \times (N/2)$ quadrants and recurse on the
8 quadrant products above, **in this exact order**: for $C_{11}$ do
$A_{11}B_{11}$ then $A_{12}B_{21}$; then $C_{12}$ ($A_{11}B_{12}$ then
$A_{12}B_{22}$); then $C_{21}$ ($A_{21}B_{11}$ then $A_{22}B_{21}$); then
$C_{22}$ ($A_{21}B_{12}$ then $A_{22}B_{22}$). A quadrant is a *view*:
track its own top-left byte address, but keep addressing every element
with the *original* matrix's row stride $N$ — quadrant-local $(r, c)$ is
at `quadrant_base + (r*N + c)*4`. When $N \le 8$, stop recursing and fall
back to the exact same triple-loop touch pattern as `naive_matmul`, run
over just that $N \times N$ sub-block (still addressed with the
full-matrix stride). You'll need your own recursive helper taking the
stride as an extra parameter — write it in `solve.cpp`; only the two
signatures above are called from outside.

## Example

For $N = 64$: `recursive_matmul` first splits into four $32 \times 32$
quadrant-products (8 recursive calls carrying the products above), each of
which splits into four $16 \times 16$'s, each of which splits into four
$8 \times 8$'s — which are small enough ($\le 8$) to run the direct
triple loop, still addressing every element with stride $64$, e.g. row 1
of an $8\times8$ block starting at byte offset `off` begins at
`off + 64*4`, not `off + 8*4`.

## What the gate checks

The driver multiplies three $64\times64$ float matrices (16384 bytes
each, 49152 bytes together — 6x the 8192-byte cache) once with
`naive_matmul` and once with `recursive_matmul`, each against its own
fresh cache, and prints both miss counts. The grader compiles `solve.cpp`
with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed miss counts match the reference}
$$

On this fixture the naive triple loop measures **266760** misses — over
a third of its 786432 total touches miss, because `B`'s column-strided
inner loop blows past the cache's capacity on every $(i,j)$ — while the
recursive version measures **2824**: once the recursion's working set
drops under 8192 bytes, that whole sub-problem's traffic is
cache-resident, a ~94x reduction from reordering the exact same
multiply-adds.
