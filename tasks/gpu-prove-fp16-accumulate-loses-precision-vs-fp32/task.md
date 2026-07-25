## Context

fp16 (IEEE-754 half precision) has 10 explicit mantissa bits — exactly
1024 representable values per octave. For any value in `[1024, 2048)`,
that means the gap between two consecutive representable fp16 values
(the ULP) is *exactly* `1.0`. An increment smaller than half that ULP
(magnitude `< 0.5`), added to a value already in that range, rounds
right back to where it started — the addition happened, but the result
is indistinguishable from having never added anything at all.

This is exactly why mixed-precision training never accumulates the
running sum (loss, optimizer state, reduction results) in fp16: each
individual update might be tiny compared to the accumulator, and fp16's
coarse ULP at that magnitude throws most of them away. fp32 accumulation
of the identical values keeps every bit of that information.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void accumulate_precision_demo(float* out_fp32, float* out_fp16,
                                           const float* base, const float* inc, int n);
```

Single-threaded (`threadIdx.x == 0` only). `acc32 = acc16 = base[0]`;
for `i` in `[0, n)`: `acc32 = acc32 + inc[i]` (plain running sum), while
`acc16 = floorf(acc16 + inc[i] + 0.5f)` (round the intermediate sum to
the nearest integer every step — fp16's ULP at this magnitude). Write
`out_fp32[0] = acc32`, `out_fp16[0] = acc16`.

## Example

`base = 1024.0`, increments `0.3, -0.2, 0.4`: `acc32` becomes
`1024.5` — every fractional contribution survives. `acc16`: step 1,
`floor(1024.3 + 0.5) = 1024`; step 2, `floor(1023.8 + 0.5) = 1024` (using
the ROUNDED 1024, not 1024.3, as the base for the next add); step 3,
`floor(1024.4 + 0.5) = 1024`. Three real updates, zero net change.

## What the gate checks

`max_abs_err <= 1e-6` against two numpy oracles on a fixed 64-increment
stream (each increment in `[-0.4, 0.4]`, base `1024.0`): the true sum
`1025.3208576...` for `out_fp32`, and `1024.0` — completely unmoved by
64 real additions — for `out_fp16`. Rounding `acc32` too (both outputs
collapsing to `1024`), or not rounding `acc16` at all (both outputs
matching the true sum), each fail one of the two oracle comparisons.
