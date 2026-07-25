## Context

A batch of ragged (variable-length) rows — different sequence lengths in
a batch, different neighbor counts per node, whatever the source — is
often handled by padding every row out to the longest one (`MAXLEN`) and
launching one thread per padded slot. That's simple, but it launches
`R * MAXLEN` threads when only `sum(lens)` of them do real work — every
padded slot is a thread (and, at the warp level, potentially a whole
*warp*) that exists purely to be idle.

The alternative: concatenate every row's REAL elements back-to-back with
no gaps, and launch exactly `sum(lens)` threads — the true amount of
work, not the worst-case rectangle around it. The cost of dropping the
padding is that a flat thread id `tid` no longer tells you which row
you're in for free (`tid / MAXLEN` doesn't work when there's no MAXLEN
stride anymore) — you have to look it up.

## Task

Implement

```cuda
__global__ void ragged_process(float* out, const float* data, const int* offsets,
                                const float* row_scale, int R, int total);
```

`data[0..total)` holds every row's real elements concatenated, row `r`
occupying `[offsets[r], offsets[r+1])` (`offsets[R]` isn't stored;
row `R-1` runs to `total`). For `tid = blockIdx.x*blockDim.x +
threadIdx.x`, guarded by `tid < total`: find `tid`'s row by scanning
every `r` in `[0, R)` and keeping track of the LAST `r` for which
`offsets[r] <= tid` (no `break` in this dialect — just keep overwriting
a `row` variable as the scan runs; `offsets` is non-decreasing, so the
last one that still qualifies is correct). Then
`out[tid] = data[tid] + row_scale[row]`.

## Example

`lens = [16, 4, 8, 2, 16, 6, 10, 3]` (`R=8`), `offsets = [0, 16, 20, 28,
30, 46, 52, 62]`, `total = 65`. Thread `tid = 25`: scanning offsets, the
last one `<= 25` is `offsets[2] = 20`, so `row = 2` — correct, since row
2 owns flat positions `[20, 28)` and `25` falls inside that range.

## What the gate checks

`check.py` seeds a fixed ragged batch (8 rows, lengths `16,4,8,2,16,6,
10,3`, summing to `65`), parses `solve.cu`, and launches
`ragged_process` with exactly `65` threads (the real element count, not
`R*MAXLEN=128`). It compares `out` against a numpy oracle built the same
way (each row's real elements plus that row's scale, concatenated) and
requires

$$
\mathrm{max\_abs\_err} = \max |{\text{out} - \text{oracle}}| \le 10^{-6}
$$

For context, `check.py` also runs a fixed, always-correct padded-launch
baseline (`R*MAXLEN=128` threads, one per padded slot) through the same
simulator and reports `warp_ratio` — the padded launch needs `4` warps,
the compact one needs `3` (`ceil(65/32)`), a genuine `25%` reduction in
warps launched for identical real work. An empty kernel body leaves
`out` all zeros, missing every row's actual values, and fails the
correctness gate immediately.
