## Context

Three families of GPU aggregation patterns need three different amounts
of synchronization:

- **Embarrassingly parallel**: every output element comes from exactly
  one thread's own data, and no two threads ever write to the same
  place. No synchronization at all — not even a barrier.
- **Atomics**: multiple threads may write to the *same* output slot, and
  which threads collide is **data-dependent** (e.g. a histogram: the bin
  a value lands in depends on the value itself, so the set of colliding
  writers can't be worked out ahead of time). An atomic read-modify-write
  is the only way to avoid lost updates.
- **Two-pass**: no data-dependent collisions, but the aggregate needs to
  combine results from **more than one block**. There is no way for one
  block to wait on another mid-kernel (no global barrier exists), so the
  combine has to happen in a *second* kernel launch, after the first one
  finishes and every block's partial result is visible.

These aren't independent: a `atomicAdd` on **device global memory**
already reaches every block in the grid, not just the threads of one
block. So if a pattern already needs atomics for its data-dependent
collisions, it does **not** additionally need a separate two-pass
combine — the same atomic pass already covers the cross-block case for
free. Atomics takes priority.

## Task

Implement

```cpp
__global__ void classify_sync_strategy(float* out, const float* cross_block,
                                        const float* shared_target_unknown, int n);
```

For every `i` in `[0, n)`, `cross_block[i]` is `1.0` if the pattern needs
a combine across multiple blocks (`0.0` otherwise), and
`shared_target_unknown[i]` is `1.0` if the write target is data-dependent
(`0.0` otherwise). Write to `out[i]`:

- `shared_target_unknown[i] > 0.5` → `out[i] = 1.0` (atomics) — checked
  **first**
- else `cross_block[i] > 0.5` → `out[i] = 2.0` (two-pass)
- else → `out[i] = 0.0` (embarrassingly parallel)

## Example

`cross_block[i]=1.0, shared_target_unknown[i]=1.0` (a histogram spread
across many blocks): `out[i] = 1.0` (atomics), **not** `2.0` — a single
kernel with `atomicAdd` on the global histogram array already handles
every block, so no separate two-pass combine is needed on top of it.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
the software GPU over all 4 combinations of the two flags (doubled to
fill 8 threads), comparing `out` against a reference computed with the
same priority rule (`np.where`, never a hardcoded label list). It
requires `max_abs_err <= 1e-9`. Checking `cross_block` before
`shared_target_unknown` — the more "obvious" reading order — gets the
`(cross_block=1, shared_target_unknown=1)` case wrong (`2.0` instead of
the correct `1.0`), since it never re-checks `shared_target_unknown`
once the `cross_block` branch already returned two-pass.
