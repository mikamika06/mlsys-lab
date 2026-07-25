## Context

Tensor cores multiply `fp16` (or `bf16`/`tf32`) inputs but accumulate
the running sum in `fp32`. This isn't a minor implementation detail —
it's the whole point of "mixed precision": the *inputs* only need
enough range and precision to represent individual activations and
weights reasonably (`fp16`'s 10 mantissa bits are plenty for that), but
a matmul's accumulator adds up many products in sequence, and rounding
*that* running total down to `fp16` after every single addition
compounds rounding error across the whole contraction. Keeping the
accumulator at full precision costs nothing extra in output storage —
only the final result gets written back — while avoiding that
compounding entirely.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void mixed_precision_matmul(const float* A, const float* B, float* C,
                                        int N, int accumulate_fp16);
```

One thread per output element. For each `k`, round **both** `A[row][k]`
and `B[k][col]` to fp16's 10-bit mantissa before multiplying — always,
regardless of `accumulate_fp16` — using
`sign(v) * round(|v|/scale) * scale` with
`scale = 2^(floor(log2|v|) - 10)`. Add the rounded product into `acc`.
If `accumulate_fp16 > 0`, **also** round `acc` itself to fp16 mantissa
right after that addition (same formula, guarding `acc == 0`); if
`accumulate_fp16 == 0`, leave `acc` at full precision. Write `acc` to
`C[idx]` once the loop over `k` finishes.

## Example

Rounding `v = 3.3` to fp16 mantissa: `e = floor(log2(3.3)) = 1`,
`scale = 2^(1-10) = 2^{-9}`, `round(3.3/2^{-9}) * 2^{-9}` lands on the
nearest fp16-representable value near `3.3` — the input rounding is
coarse enough to matter, but a single rounding per input, not one per
accumulation step.

## What the gate checks

The grader launches `mixed_precision_matmul` twice against the same
fixed `16x16` matrices — once with `accumulate_fp16=0`, once with
`accumulate_fp16=1` — and compares both against an exact float64
`A @ B`. It requires

$$
\mathrm{fp32\_acc\_err} \le 10^{-3} \quad\text{and}\quad
\mathrm{fp16\_acc\_err} - \mathrm{fp32\_acc\_err} \ge 2\times10^{-4}
$$

The second gate matters as much as the first: a kernel that ignores
`accumulate_fp16` entirely (always keeping the accumulator at full
precision) would satisfy the first gate but show **no** difference
between the two modes — caught by the improvement floor. On this
fixture, `fp32_acc_err = 2.63\times10^{-4}$ against
`fp16_acc_err = 1.06\times10^{-3}$ — accumulating in full precision
cuts the error by about 4x, from nothing but where the rounding stops.
