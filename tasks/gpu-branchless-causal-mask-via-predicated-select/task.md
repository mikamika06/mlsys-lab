## Context

Causal attention masking is the textbook `if (j > i) score = -inf;`
check: position `i` can't attend to a future position `j`. Written as an
`if`/`else` that decides *which memory operations run*, a warp whose 32
lanes straddle the `j <= i` boundary — which happens for almost every
row once the row is more than a warp wide — has some lanes execute the
load-then-store path and others the store-only path. Real hardware
serializes those two paths one after another (predicated execution or
explicit branching, either way the warp pays for both paths' worth of
issue slots). The fix isn't to skip work more cleverly — it's to make
every lane do the *exact same sequence of memory operations* regardless
of the mask outcome, and fold the masking decision into an arithmetic
value instead.

## Task

Rewrite the branchy causal mask into a branchless (predicated) form:

```cuda
__global__ void causal_mask(float* out, const float* score, int n);
```

`idx = blockIdx.x * blockDim.x + threadIdx.x` indexes an `n x n` matrix
row-major (`i = idx / n`, `j = idx % n`). Every thread must:

1. **Always** load `score[idx]` (no `if` gating the load).
2. Compute a `0`/`1` predicate, `keep = (j <= i) ? 1.0f : 0.0f`.
3. Blend: `out[idx] = keep * v + (1.0f - keep) * neg_inf`, using
   `-1.0e30f` as the masked-out sentinel.
4. **Always** store `out[idx]` (no `if` gating the store).

Every thread issues the identical sequence of memory operations — one
load, one store — no matter which side of the mask it lands on.

## Example

For `n = 64`, thread `idx` with `i = 5, j = 3` (`j <= i`): `keep = 1.0`,
`out[idx] = score[idx]`. Thread with `i = 5, j = 9` (`j > i`):
`keep = 0.0`, `out[idx] = -1.0e30`. Both threads still load `score[idx]`
and store `out[idx]` exactly once — only the arithmetic differs, not the
access pattern.

## What the gate checks

`check.py` seeds a fixed random `64x64` score matrix, parses `solve.cu`,
and launches `causal_mask` as a `64x64`-thread grid (one thread per
matrix entry) on the software GPU. It compares the output against a
numpy oracle (`np.where(j <= i, score, -1e30)`, computed from the same
seeded input) and reads the simulator's own `divergences` count — the
number of warps whose 32 lanes issued a *different number* of global
accesses, this simulator's proxy for control-flow divergence. It
requires

$$
\mathrm{max\_abs\_err} \le 10^{-6} \quad \text{and} \quad \mathrm{divergences} = 0
$$

The reference passes both (`max_abs_err=0.0`, `divergences=0`). A
value-correct but branchy version — `if (j <= i) out[idx]=score[idx];
else out[idx]=-1e30;`, which skips the load in the `else` case — gets the
values exactly right (`max_abs_err=0.0`) but reports `divergences=62`:
almost every one of the 128 warps in a 64-wide row straddles the `j<=i`
boundary somewhere, and its lanes end up with mismatched access counts.
