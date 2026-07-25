## Context

Reading `n` floats one at a time (`float1`, one load instruction per
element) issues `n` load instructions. Reading them 4-at-a-time
(`float4`, one load instruction moves 4 elements) issues only
$\lfloor n/4 \rfloor$ vectorized loads for the bulk — but if `n` isn't a
multiple of 4, the last `n % 4` elements don't fill a whole `float4` and
have to fall back to individual scalar loads. Total vectorized-path
instruction count:

$$\text{loads\_float4} = \left\lfloor \frac{n}{4} \right\rfloor + (n \bmod 4)$$

The instruction-count reduction factor vectorizing buys you is just
$\text{loads\_float1} / \text{loads\_float4} = n / \text{loads\_float4}$
— close to $4\times$ when `n` is a multiple of 4 (no tail at all), and
noticeably less than $4\times$ the closer the tail gets to 3 extra scalar
loads.

## Task

Write a CUDA-C kernel (single thread — this is pure arithmetic, there is
no `float4` type in this language subset to actually load with):

```cpp
__global__ void float4_instr_counts(float* out, int n);
```

Compute and write:

- `out[0] = loads_float1` = `n`
- `out[1] = loads_float4` = `n / 4 + n % 4`
- `out[2]` = `out[0] / out[1]`, the reduction factor

## Example

| $n$ | loads_float1 | loads_float4 | ratio |
|---|---|---|---|
| 17   | 17   | $4 + 1 = 5$    | $3.4$ |
| 100  | 100  | $25 + 0 = 25$  | $4.0$ |
| 256  | 256  | $64 + 0 = 64$  | $4.0$ |
| 4097 | 4097 | $1024 + 1 = 1025$ | $\approx 3.997073$ |

$n = 100$ and $n = 256$ are exact multiples of 4 — no tail, exactly
$4\times$ fewer instructions. $n = 17$ has a 1-element tail out of only 5
total float4-path loads, so the reduction falls well short of $4\times$;
$n = 4097$'s tail is proportionally tiny (1 extra load out of 1025), so
its ratio sits much closer to $4$.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it
(single thread) on the software GPU once for each of 4 fixed `n` values
(`17, 100, 256, 4097`), requiring `max_abs_err <= 1e-6` against
`loads_float1`, `loads_float4`, and the ratio computed directly in
Python for each. Using `n // 4` alone for `loads_float4` (dropping the
tail term entirely) gets `n = 100` and `n = 256` right by coincidence
(their tail is `0`) but is off by `1` on both `17` and `4097`, and the
ratio is wrong too — the empty starter fails on every value.
