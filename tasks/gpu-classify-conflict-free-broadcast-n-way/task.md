## Context

`__shared__` memory is split into 32 banks (one word per bank, wrapping
every 32 words: bank $= \text{word\_index} \bmod 32$). A single warp's
access to shared memory falls into one of three categories:

- **Conflict-free**: every active thread's word index lands in a
  DIFFERENT bank. Serviced in one cycle.
- **Broadcast**: every active thread reads the exact SAME word (same
  bank, same address). Real hardware detects this and services it in one
  cycle too — a broadcast is not a conflict.
- **$k$-way conflict**: some bank is hit by $k$ threads at $k$ DIFFERENT
  word indices. That bank has to serialize $k$ separate accesses.

## Task

Write a CUDA-C kernel:

```cpp
__global__ void bank_pattern(float* out, const float* in, int case_id, int n);
```

Launched as a single block of 32 threads (one warp), with `__shared__
float buf[128];`. Depending on `case_id`, thread `tid` computes an index
`idx` realizing one of the three named patterns, then
`buf[idx] = in[tid]; __syncthreads(); out[tid] = buf[idx];`:

- `case_id == 0` (conflict-free): `idx = tid` — 32 threads, 32 distinct
  banks.
- `case_id == 1` (broadcast): `idx = 0` — every thread writes and reads
  the same word.
- `case_id == 2` (4-way conflict): `idx = (tid % 4) * 32 + (tid / 4)` —
  bank $= \text{idx} \bmod 32 = \lfloor \text{tid}/4 \rfloor$, so the 4
  threads sharing a `tid / 4` value collide on one bank, each at a
  DIFFERENT word.

## Example

The grader relaunches your kernel once per `case_id` with the same 32
random inputs. For `case_id = 0`, `out[tid]` reads back exactly `in[tid]`
(each thread wrote and re-read its own private word) — 2 shared-memory
"waves" (one for the write step, one for the read-back step, both already
conflict-free). For `case_id = 2`, the *values* come out identically
correct (`out[tid] == in[tid]` again — a bank conflict never corrupts
data, it only serializes the hardware), but the wave count jumps to 8: 4x
the conflict-free case, one "wave" per colliding thread at both the write
and the read step.

For `case_id = 1` (broadcast), every thread's `idx` is the SAME address,
so the LAST thread the simulator executes for that phase — thread 31 —
is the one whose write survives (this simulator executes a block's
threads in `tid` order within each phase, so this is fully
deterministic): every `out[tid]` reads back `in[31]`, not `in[tid]`
individually, and the wave count stays at 2 — a broadcast, not a
conflict.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it
thread-by-thread on the software GPU for all 3 cases, checking both the
output values (`max_abs_err <= 1e-9` against the reference pattern's
result for each case: per-thread values for cases 0 and 2, thread 31's
value broadcast for case 1) and the exact `smem_waves` count per case
(`waves_exact == 1.0`: must be `2, 2, 8` for cases `0, 1, 2`
respectively). Getting case 2's *values* right without actually
colliding 4 threads per bank (e.g. reusing case 0's `idx = tid`) passes
the value check but reports `smem_waves = 2` instead of `8`, and fails
`waves_exact`. The empty starter never writes `out` at all and fails
both gates.
