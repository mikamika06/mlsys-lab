## Context

In a tiled matmul-style kernel computing an $M \times N$ output over a
depth-$K$ contraction, the simplest launch gives every output element
its own thread: $M \times N$ threads, each looping over $K$, each
independently loading its own row value from $A$ and column value from
$B$ at every step. That's $M N K$ loads from $A$ *and* $M N K$ loads
from $B$ — but every thread computing output $(i, j)$ loads exactly the
same $A_{i,k}$ that every *other* thread computing $(i, j')$ for a
different $j'$ also loads. That's pure redundancy: the same value,
fetched from global memory once per thread, when one fetch could serve
them all.

**Register blocking** (thread coarsening) fixes this by giving each
thread $C$ *adjacent* output columns instead of $1$. Now there are only
$M \times (N/C)$ threads — and at each step $k$, a thread loads
$A_{i,k}$ into a register **once** and reuses it across all $C$ of its
outputs, instead of $C$ separate threads each loading it independently.
$B$ doesn't get this benefit: every output still needs its own,
different, column value of $B$, coarsening or not.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void derive_loads(int M, int N, int K, int C, float* out);
```

Write into `out[0]` the total number of loads from $A$ **with**
register blocking: `threads = M * (N / C)`, each loading $K$ times —
`out[0] = threads * K`. Write into `out[1]` the total number of loads
from $B$, which register blocking does not change:
`out[1] = M * N * K`.

## Example

$M=16, N=64, K=32, C=4$: `threads = 16 * (64/4) = 256`, so
`out[0] = 256 * 32 = 8192`. `out[1] = 16 * 64 * 32 = 32768`. The ratio
`out[1] / out[0] = 4` — exactly $C$: coarsening by 4 cuts $A$'s traffic
to a quarter, no matter what $M$, $N$, or $K$ are.

## What the gate checks

The grader launches `derive_loads` for 5 fixed `(M, N, K, C)` scenarios
(all with `N` evenly divisible by `C`) and compares both outputs
against an independently computed oracle. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{both outputs match the oracle on every one of the 5 scenarios}
$$

Across all 5, `out[1] / out[0]` comes out to exactly `4, 8, 2, 16, 5` —
matching each scenario's own `C` precisely, regardless of how different
`M`, `N`, and `K` are between them. A formula that gets the thread
count wrong (e.g. forgetting to divide `N` by `C`, or applying the
reduction to `B` too) fails at least one of the 5 comparisons.
