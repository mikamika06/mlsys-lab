## Context

Matrix multiplication computes

$$C_{ij} = \sum_{k=0}^{n-1} A_{ik} B_{kj} .$$

The mathematical result is independent of loop order, but the memory
access pattern is not. Over row-major arrays, loop order changes which
addresses get reused before a cache line is evicted. Consider the three
loop nests over the same $(i, j, k)$ index cube:

$$
\text{ijk}: i \to j \to k
\qquad
\text{ikj}: i \to k \to j
\qquad
\text{jki}: j \to k \to i .
$$

For `ijk`/`ikj`, the innermost loop sweeps one row of `A`/`B` at a time --
contiguous, cache-friendly. For `jki`, the innermost loop sweeps `i`
while `j` and `k` are fixed: every step jumps a full row's stride in
both `A` and `C`, touching a fresh cache line almost every time.

## Task

Implement

```cpp
void rank_matmul_orders(int n, char out[3][4]);
```

Using the fixed cache probe from `sol.hpp` (`cache_reset()`, `touch(addr)`,
`cache_misses()` -- a 64-byte-line, 32-set, 2-way LRU model), run THREE
separate passes of `C += A * B` over the same `n x n x n` index space, one
per loop nest above, each starting from a fresh `cache_reset()`. In every
pass, for each `(i, j, k)` touch `a_addr(n,i,k)`, then `b_addr(n,k,j)`,
then `c_addr(n,i,j)` -- exactly once each -- while the loop nesting
follows the stated order. After each pass, read `cache_misses()`.

Write `"ijk"`, `"ikj"`, `"jki"` into `out[0]`, `out[1]`, `out[2]`, sorted
from FEWEST misses to MOST. Break ties by the fixed priority
`ijk < ikj < jki`.

## Example

With `n = 24` (three 24x24 double matrices = 13824 bytes, well over the
model cache's 4096-byte capacity), the reference measures `ijk=1097`,
`ikj=1120`, `jki=3613` misses -- so the correct ranking is
`{"ijk", "ikj", "jki"}`.

## What the gate checks

`exact_match` on the printed ranking for `n=24`. Running the loop nests
in the wrong order (e.g. swapping the `ikj` and `jki` bodies), touching
`a_addr`/`b_addr`/`c_addr` in the wrong order, forgetting `cache_reset()`
between passes (letting one pass's misses leak into the next), or getting
the tie-break priority backwards, all change the printed ranking.
