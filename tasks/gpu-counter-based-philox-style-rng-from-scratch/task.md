## Context

A classic LCG-style RNG carries state forward one draw at a time: to get
value 1000 of the stream, you must have already produced values 0
through 999. That's fundamentally serial — useless for a GPU, where
thousands of threads need a random number *simultaneously*, each for its
own purposes (dropout masks are the textbook case: every element of every
layer needs an independent random draw, all in one kernel launch).

**Counter-based** RNGs (Philox and Threefry are the well-known real
ones) sidestep this entirely: instead of state, each output is a pure
*function* of a shared `key` and that output's own `counter` — no draw
depends on any other. Thread $i$ can compute draw $i$ with zero
communication, in any order, on any device, and get the exact same answer
every time.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void philox_style_rng(float* out, const float* counters, float key, int n);
```

For thread `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
start with `x = counters[i]`, then run 3 mixing rounds, `r = 0, 1, 2`:

$$x \leftarrow \big(x \cdot 48271 + \text{key} + r \cdot 7919\big) \bmod 1000003$$

and write `out[i] = x / 1000003` (normalized into `[0, 1)`).

## Example

`key=12345`, `counter=0`: round 0 gives `x = (0*48271 + 12345 + 0*7919)
mod 1000003 = 12345`. Round 1: `x = (12345*48271 + 12345 + 7919) mod
1000003 = 923974`. Round 2: `x = (923974*48271 + 12345 + 15838) mod
1000003 = 43334`. Final: `out[0] = 43334 / 1000003 ≈ 0.0433339` — pure
arithmetic throughout, reproducible bit-for-bit by any correct
implementation.

## What the gate checks

`max_abs_err <= 1e-9` on all 64 outputs (`counters[i] = i`, fixed
`key=12345`), compared against a numpy oracle running the identical
3-round formula. Using the wrong round count, applying `key` only once
instead of every round, forgetting the per-round `r * 7919` term (making
every round identical), or normalizing by the wrong modulus, all diverge
from the reference stream.
