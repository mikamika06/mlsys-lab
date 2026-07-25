## Context

Triton-style kernels compute each program's (block's) share of the work
from `program_id` — `offset = program_id * BLOCK + local_index` — the same
math CUDA writes as `i = blockIdx.x*blockDim.x + threadIdx.x`. When the
data size isn't a multiple of `BLOCK`, that offset runs past the real data
in the launch's last block, and a **mask** (`offset < n`) has to guard
every memory operation that touches it — but different operations need the
mask applied differently. A *store* to an output sized exactly to the
launch (padded past `n`) doesn't need masking at all — every thread has a
valid slot to write to. A *load* from an input sized exactly `n`, with no
padding, absolutely does: reading past its end reads whatever memory
happens to sit next, and Triton's `tl.load(ptr, mask=mask, other=0.0)`
exists precisely to replace that undefined read with a defined default.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void masked_scale_fill(float* out, const float* in, int n, float s);
```

`out` is sized to the full launch (`blockDim.x * gridDim.x`, padded past
`n`); `in` has exactly `n` real elements. For `i = blockIdx.x*blockDim.x +
threadIdx.x`: load `v = in[i]` only when `i < n`, otherwise use `0.0`
(never read `in[i]` for `i >= n`); then store `out[i] = s * v`
unconditionally — every thread has a valid slot in `out`, so the tail
threads correctly write `s * 0.0 = 0.0` there.

## Example

`n = 90`, `blockDim.x = 32`, `gridDim.x = 3` (96 threads total) — the last
6 threads (`i = 90..95`) are past `n`. Immediately after `in`'s 90 real
elements in memory sits a "trap" region filled with `777.0`. A kernel that
reads `in[i]` unconditionally (no mask on the load) reads straight into
that trap for `i = 90..95` and stores `s * 777.0` there instead of the
required `0.0` — visibly, obviously wrong, not a crash.

## What the gate checks

`check.py` builds the fixture (with the trap region right after `in`),
parses `solve.cu`, and runs `masked_scale_fill` on the software GPU
(`arena.cuda_sim.GPU`) with a 3-block, 32-thread launch over `n = 90`. It
requires `max_abs_err == 0.0` against a reference that fills indices
`[0, 90)` with `s * in[i]` and indices `[90, 96)` with exactly `0.0` —
comparing the full 96-element `out` buffer, not just the first 90 elements,
is what makes an unmasked load's leak visible.
