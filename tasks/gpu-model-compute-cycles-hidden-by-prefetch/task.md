## Context

Processing $T$ tiles, each needing a load (latency $L$) then compute
($C$ cycles), takes $T(L+C)$ cycles run serially — nothing overlaps.
Prefetching tile $i{+}1$'s load while tile $i$ computes changes that: the
very first load ($L$) still has to happen alone, and the very last tile's
compute ($C$) still has to happen alone, but every load/compute pair in
between runs *concurrently*, taking whichever of the two is longer:

$$\text{overlap}(T,L,C) = L + (T-1)\cdot\max(L,C) + C$$

The gap between the two — how many cycles the pipelining actually
*hides* — is $(T-1)\cdot\min(L,C)$: at every one of the $T-1$ overlap
opportunities, the *shorter* of load and compute disappears completely
into the longer one.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void pipeline_cycles(float* out_serial, float* out_overlap,
                                 const float* T, const float* L, const float* C, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out_serial[i] = T[i] * (L[i] + C[i])`, `out_overlap[i] = L[i] + (T[i] -
1) * fmaxf(L[i], C[i]) + C[i]`.

## Example

`T=10, L=200, C=500` (compute-bound — loads always fit inside compute):
`serial = 10*(700) = 7000`; `overlap = 200 + 9*500 + 500 = 5200`. Compare
`T=10, L=500, C=200` (load-bound, same totals swapped): `serial = 7000`
(unchanged — same total work), `overlap = 500 + 9*500 + 200 = 5200`
(identical, since `max(500,200)=500` either way — pipelining hides
whichever resource is *shorter*, and the shorter one flips between the
two scenarios along with everything else).

## What the gate checks

`max_abs_err <= 1e-3` on 5 fixed `(T, L, C)` scenarios, including
`T=1` (a single tile: no overlap opportunity exists, so `overlap` must
equal `serial` exactly) and `L == C` (the tie case, `max(L,C)` still well
defined). Using `min` instead of `max` in the overlap formula, or
computing `overlap` without the standalone leading `L` and trailing `C`
terms, gets every multi-tile scenario wrong while `T=1` might still
coincidentally pass — this task's fixed set is chosen so an off formula
fails at least one of the 5.
