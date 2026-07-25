## Context

`fp16` and `bf16` are both 16-bit floats, but they split their bits
differently. `fp16` spends 10 bits on mantissa and only 5 on exponent —
great precision, but its largest finite value is a modest
$65504 = (2 - 2^{-10}) \times 2^{15}$. `bf16` spends only 7 bits on
mantissa but keeps `fp32`'s full 8-bit exponent — coarser precision, but
its dynamic range matches `fp32` almost exactly, topping out around
$3.3895314\times10^{38} = (2 - 2^{-7}) \times 2^{127}$, over 33 orders
of magnitude higher than `fp16`'s ceiling.

This is exactly why mixed-precision training on tensor cores usually
prefers `bf16` over `fp16` for accumulators and activations: a value
that would silently become `inf` in `fp16` (gradient spikes, unnormalized
logits, large attention scores) is still perfectly representable in
`bf16`, at the cost of a bit more rounding error per value, not a
correctness cliff.

## Task

Implement the kernel `classify_overflow` in `solve.cu`:

```cuda
__global__ void classify_overflow(const float* x, float* out, int n);
```

For each `i < n`, let `v = fabsf(x[i])`. Set `out[i] = 1.0` if `v`
**overflows `fp16`** ($v > 65504$) **but is still within `bf16`**
($v \le 3.3895314 \times 10^{38}$); otherwise set `out[i] = 0.0`.

## Example

$x = 70000$: overflows `fp16` (`70000 > 65504`) and is far below
`bf16`'s ceiling — `out = 1.0`. $x = -30000$: `fabsf` gives `30000`,
which doesn't overflow `fp16` at all — `out = 0.0`. $x = 5 \times
10^{40}$: overflows `fp16`, but *also* overflows `bf16`'s ceiling —
`out = 0.0`, since the value isn't representable in either format.

## What the gate checks

The grader parses `solve.cu` with the CUDA-C frontend, runs
`classify_overflow` on a fixed 64-value input (a seeded mix of signs
and magnitudes spanning $10^{-10}$ to $10^{40}$, plus explicit values
sitting right at and around each format's boundary), and compares the
output against a numpy oracle computed directly from
`FP16_MAX = 65504.0` and `BF16_MAX = 3.3895313892515355 \times
10^{38}$. It requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 64 outputs matches the oracle exactly}
$$

Getting this right means comparing against *both* thresholds correctly
in the *same* direction (`>` for the fp16 ceiling being crossed, `<=`
for the bf16 ceiling not being crossed) — swapping either comparison,
or checking only one of the two formats, mis-classifies a large
fraction of the 64 test values.
