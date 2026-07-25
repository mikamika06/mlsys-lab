## Context

An elementwise kernel over a flat array only needs `out[tid] = f(in[tid])`
— there's no row or column to speak of. But it's easy to inherit
row/column bookkeeping from nearby matrix code and apply it where it
doesn't belong: re-derive `row = tid / w`, `col = tid % w` from the flat
thread id, then re-flatten with the two swapped — `idx = col * h + row`
instead of `idx = row * w + col` (which is just `tid` again). The
computation is still a correct bijection over every valid index (every
element still gets touched by exactly one thread, so the final array
content is byte-for-byte identical either way) — it just assigns
consecutive THREADS to indices `h` apart instead of `1` apart. A whole
warp that should coalesce into a single memory transaction instead
scatters across up to 32 different segments.

## Task

Your starting point in `solve.cu` computes `out[idx] = 2*in[idx] + 1` for
a `w * h`-element flat array, but derives `idx` through an accidental
transpose:

```cpp
int row = tid / w;
int col = tid % w;
int idx = col * h + row;   // BUG
```

Fix it: there is no matrix here, just a flat array — `out[tid] =
2.0f * in[tid] + 1.0f;` directly. Consecutive threads then touch
consecutive addresses, which is all "coalesced" ever means.

## Example

With `w = 32, h = 8` (`n = 256`) launched as one warp per block
(`block = 32`), the fixed kernel's single warp touches 32 consecutive
elements per access step — 1 segment per step, 1 read + 1 write per warp
= `16` transactions total across all 8 warps. The broken version's `idx =
col * h + row` scatters those same 32 threads' addresses `h = 8` apart —
spanning the entire 256-element range within a single warp — for `128`
transactions: 8x worse, for the exact same final array contents
(`max_abs_err = 0` either way).

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over a fixed 256-element fixture, requiring `max_abs_err <=
1e-9` AND `transactions <= 20` against the simulator's own measurements.
The broken starter passes correctness outright (`max_abs_err = 0` — the
bug never touches the wrong VALUES, only which thread computes which
one) but reports `transactions = 128`, far past the threshold — proof
that "the output is right" and "the access pattern is coalesced" are two
separate things this gate checks separately.
