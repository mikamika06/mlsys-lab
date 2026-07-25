## Context

The attention mechanism's $QK^T$ contraction multiplies a query matrix
$Q \in \mathbb{R}^{S \times d}$ by the transpose of a key matrix
$K^T \in \mathbb{R}^{d \times S}$ to produce a score matrix

$$
\text{score}_{ij} = \sum_{k=0}^{d-1} Q_{ik} \cdot K_{jk}.
$$

Both $Q$ and $K$ are stored **row-major** (each row contiguous). A naive
$i$-$j$-$k$ loop computes every score by walking all of $j$ (all $S$ rows of
$K$) to completion for a single row $i$ of $Q$ before moving to the next
$i$ — so if $K$ doesn't fit in cache on its own, it gets evicted and
re-fetched from DRAM again for every single row of $Q$.

**Tiling** (blocking) fixes this: pick a block size $B$ and process the score
matrix in $B \times B$ tiles. Within one tile, $B$ rows of $Q$ and $B$ rows
of $K$ are reused across each other's inner loop before the tile is done and
the next one starts — so instead of re-streaming all of $K$ once per row of
$Q$, you re-stream roughly $1/B$ as much of it per row.

## Task

Implement

```cpp
void qkt_access(int S, int d, int B, int elem_bytes);
```

which, for every $(i, j, k) \in [0,S) \times [0,S) \times [0,d)$ exactly
once, calls `touch()` (declared in `sol.hpp`, a cache-access hook the
driver defines) on the byte address of $Q[i,k]$ then of $K[j,k]$:

$$
Q[i,k] \text{ addr} = (i \cdot d + k) \cdot \text{elem\_bytes}
$$
$$
K[j,k] \text{ addr} = S \cdot d \cdot \text{elem\_bytes} + (j \cdot d + k) \cdot \text{elem\_bytes}
$$

Tile the $i$ and $j$ loops in blocks of $B$ (outer loops step $ii$, $jj$ by
$B$; inner loops range $i \in [ii, ii{+}B)$, $j \in [jj, jj{+}B)$, $k \in
[0,d)$) instead of iterating $i$ and $j$ over their full range directly.

## Example

With $S=4, d=4, B=2$: the naive order finishes all $j \in [0,4)$ for $i=0$
before touching $i=1$ — every row of $Q$ re-touches every row of $K$ from
scratch. The tiled order instead finishes tile $(ii,jj)=(0,0)$ — $i,j \in
\{0,1\}$, all $k$ — before moving to $(ii,jj)=(0,2)$, reusing $K$'s rows 0-1
across both rows 0-1 of $Q$ within that tile.

## What the gate checks

`main.cpp` runs `qkt_access(S=128, d=64, B=16, elem_bytes=4)` against a
fixed single-level, set-associative LRU cache model (64-byte lines, 64
sets, 8-way — 32768 bytes total), and prints the resulting miss count. At
these sizes $Q$ and $K$ are 32768 bytes each — together twice the size of
the whole cache — so the loop order has a real, measured effect: the
reference (tiled) order measures **4608** misses, while the naive
(unblocked) $i$-$j$-$k$ order measures **9124** — almost double, since it
re-streams all of $K$ from DRAM on nearly every row of $Q$. `verify_native.sh`
compiles `solve.cpp` and `ref.cpp` against the same `main.cpp` with
`clang++ -O2 -std=c++20` and requires the printed miss count to match the
reference exactly.
