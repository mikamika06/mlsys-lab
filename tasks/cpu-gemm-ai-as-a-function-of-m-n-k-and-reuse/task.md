## Context

The roofline model needs a kernel's **arithmetic intensity** — FLOPs done
per byte moved from memory — to predict whether it's compute-bound or
memory-bound. For GEMM, that number isn't fixed: it depends on *how* the
computation is tiled, because tiling controls how many times each operand
byte gets reused before it's evicted and has to be re-fetched.

Consider $C_{M \times N} = A_{M \times K} \times B_{K \times N}$, computed
one $T \times T$ output **tile** at a time (each tile accumulates over the
full $K$-deep reduction before moving to the next tile). Within a tile:

- every element of the $A$-panel it touches is loaded once and reused for
  all $T$ output columns of that tile,
- every element of the $B$-panel it touches is loaded once and reused for
  all $T$ output rows of that tile.

So $A$ (all $M \times K$ elements of it) gets re-read once per
column-tile — $N/T$ times in total — and $B$ gets re-read once per
row-tile — $M/T$ times in total. $C$ is written exactly once.

$$\text{FLOPs} = 2MNK \qquad \text{(one multiply + one add per output element per } k \text{ step)}$$

$$\text{bytes} = \text{elem\_bytes} \left( MK\cdot\frac{N}{T} \;+\; KN\cdot\frac{M}{T} \;+\; MN \right)$$

$$\text{AI}(M,N,K,T) = \frac{\text{FLOPs}}{\text{bytes}}$$

At $T = 1$ this is the fully-naive baseline (every operand re-read for
every single FMA — memory-bound). As $T$ grows, the redundant re-reads
shrink and AI climbs toward compute-bound territory.

## Task

Implement

```cpp
double gemm_arithmetic_intensity(long M, long N, long K, long tile, long elem_bytes);
```

exactly per the formula above. `tile` always evenly divides both `M` and
`N` in the driver's scenarios.

## Example

For $M=N=K=256$, `elem_bytes=4`: at `tile=1`, $\text{AI} \approx 0.2495$
(barely above the $1/\text{elem\_bytes}$ floor — almost every byte moved
buys under one FLOP). At `tile=128`, $\text{AI} = 25.6$ — over 100x
higher, purely from reuse, with the *same* $M,N,K$ and the *same* total
FLOP count.

## What the gate checks

`max_abs_err <= 1e-6` on six fixed `(M,N,K,tile,elem_bytes)` scenarios
(one shape swept across four tile sizes to show the reuse trend, plus two
more shapes/element sizes). Dropping the `elem_bytes` factor, using `M*N*K`
instead of `2*M*N*K` for FLOPs, forgetting to divide by `tile`, or reusing
`A`'s and `B`'s tile-count formulas for the wrong operand, all shift the
printed numbers well past the tolerance.
