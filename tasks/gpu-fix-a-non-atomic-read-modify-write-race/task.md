## Context

256 threads each need to add `1.0` to one shared counter. Without an
atomic add, "read the counter, add 1, write it back" is only safe if
exactly one thread ever does it. The moment more than one thread's read
happens before any of their writes, every one of those threads computed
its "+1" from the *same* stale value — and whichever write lands last is
the only increment that survives. `__syncthreads()` between the read and
the write doesn't fix this; it makes it **worse**, by *guaranteeing*
every thread's read finishes before any thread's write starts, turning a
race that might occasionally get lucky into one that loses updates every
single time.

This simulator, like the CUDA-C subset it models, has no `atomicAdd`.
The real fix isn't more synchronization around the shared accumulate —
it's avoiding the shared accumulate entirely: give every thread its own
private slot to write into (so no two threads ever target the same
address), then combine all the private slots with an ordinary
barrier-synchronized tree reduction, which is race-free because at every
step of the reduction, each active address is read and written by
exactly one thread.

## Task

`solve.cu` computes `race_free_count`, meant to return `256.0` (one
count per thread) in `out[0]`, but has exactly the bug described above:
every thread reads a single shared `counter[0]`, a barrier forces every
read to finish before any write, then every thread computes and writes
back `counter[0] + 1`. Fix it by replacing the shared single-cell
accumulate with:

1. Every thread writes `1.0f` into its **own** slot,
   `sdata[threadIdx.x]`, of a `256`-element `__shared__` array.
2. `__syncthreads();`
3. A standard tree reduction: for `stride = 128, 64, 32, ..., 1`, if
   `threadIdx.x < stride`, `sdata[threadIdx.x] += sdata[threadIdx.x +
   stride];`, followed by `__syncthreads();` after every step.
4. Thread `0` writes `out[0] = sdata[0];`.

## Example

With only 4 threads (`stride` starting at 2 for illustration): each
writes `1.0` into its own slot, `sdata = [1,1,1,1]`. Step `stride=2`:
threads `0,1` compute `sdata[0]+=sdata[2]`, `sdata[1]+=sdata[3]` →
`[2,2,1,1]`. Step `stride=1`: thread `0` computes `sdata[0]+=sdata[1]` →
`[4,2,1,1]`. `out[0] = 4.0` — every thread's contribution counted
exactly once, with no two threads ever touching the same address at the
same step.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it
with 256 threads, comparing `out[0]` against `256.0`. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

The shipped `solve.cu` measures `out[0] = 1.0` — `max_abs_err = 255.0` —
every one of the 256 threads' increments except one is silently lost.
