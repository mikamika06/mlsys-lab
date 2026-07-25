## Context

A kernel's launch configuration — how many blocks, how many threads per
block, and how work gets mapped from global thread id to output index —
leaves fingerprints in its output, even without seeing the launch call
itself. Three common mappings for filling an array of `n = 64` elements
with `blockDim = 32`:

- **flat**: one thread per output, `gridDim = 2` (`64/32`). Thread `t`
  writes `out[t] = t` — the identity, since each thread's global id
  *is* its output index.
- **grid-stride**: fewer threads than outputs, `gridDim = 1` (only 32
  threads total). Each thread loops, writing multiple outputs spaced
  `32` apart; the value each thread writes (its own id) repeats with
  period `32`, so `out[t] = t \bmod 32`.
- **2d (transposed)**: `gridDim = 2` again, but indices assigned in
  column-major order across an `8x8` tile instead of the natural
  row-major order — `out[t] = (t \bmod 8) \times 8 + \lfloor t/8 \rfloor`.

Each mapping produces a distinct, recognizable pattern in the output
array, entirely independent of what the kernel's *payload* computation
was.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void reconstruct_launch(const float* obs, int n, float* result);
```

For every `t` in `[0, n)`, compute what `obs[t]` *would* be under each
of the three hypotheses above, and count mismatches against the actual
`obs[t]` for each. Exactly one hypothesis will have zero mismatches on
any valid input. Write `result[0] = mapping_kind` (`0` for flat, `1`
for grid-stride, `2` for 2d) and `result[1] = gridDim` (`2`, `1`, `2`
respectively) for whichever hypothesis matched.

## Example

`obs = [0, 1, 2, ..., 63]`: matches the flat hypothesis exactly
(`obs[t] == t` for every `t`) — `result = [0, 2]`. `obs = [0, 1, ...,
31, 0, 1, ..., 31]`: fails the flat check at `t=32` (`obs[32]=0 != 32`)
but matches grid-stride everywhere (`t % 32` reproduces exactly this
repeating pattern) — `result = [1, 1]`.

## What the gate checks

The grader constructs all three observed arrays independently (flat,
grid-stride, and the transposed-2d pattern) and launches
`reconstruct_launch` once per array, comparing both outputs against the
correct `(mapping_kind, gridDim)` pair for that array. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{both recovered values are correct on all 3 observed arrays}
$$

Getting one hypothesis right in isolation is easy — the real test is
distinguishing all three correctly from the *same* piece of code,
including correctly recognizing the grid-stride case's telltale
repeating period and the 2d case's transposed (not simply reordered)
index pattern.
