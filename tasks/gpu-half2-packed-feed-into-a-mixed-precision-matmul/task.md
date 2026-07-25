## Context

Tensor cores compute in **mixed precision**: inputs are stored and
multiplied in a narrow format (fp16, packed two values to a `half2` lane
so one load fetches a pair at once), but products are accumulated in full
fp32 to keep summation error from compounding across many terms. This
language subset has no real `half2`/fp16 type, so precision loss is
modeled directly with arithmetic: quantize a value to the nearest
multiple of a small step,

$$\hat{x} = \left\lfloor \frac{x}{q} + 0.5 \right\rfloor \cdot q, \qquad q = \frac{1}{256}$$

— the same rounding-to-a-grid effect fp16's limited mantissa has, without
needing real bit-level fp16 encoding.

The `half2` PACKING itself is an access-pattern fact, not a numeric one:
one lane owns a PAIR of adjacent elements (indices `2*tid` and
`2*tid + 1`), the same pair a real `half2` load would fetch in a single
instruction.

## Task

Write a CUDA-C kernel, launched as one warp (32 threads) over `n = 64`
elements:

```cpp
__global__ void half2_matmul_dot(float* out, const float* a, const float* b, int n);
```

Thread `tid` owns the pair `i0 = 2*tid`, `i1 = i0 + 1`. Quantize `a[i0]`,
`b[i0]`, `a[i1]`, `b[i1]` each to the nearest multiple of `qstep =
1.0f/256.0f`, compute this lane's partial sum `val = a0*b0 + a1*b1` in
fp32, then combine all 32 lanes' `val`s with a warp-shuffle reduction
(`__shfl_down_sync` at deltas `16, 8, 4, 2, 1`, in that order) and have
thread `0` write the final sum to `out[0]`.

## Example

On a fixed 64-element random fixture (`a, b` uniform in `[-2, 2]`), the
full-fp32 dot product is `≈ -11.454851`. Quantizing every input to the
nearest `1/256` before multiplying moves the result to `≈ -11.451859` —
off by about `0.003`, the accumulated rounding from 64 quantized
products, nowhere near enough to matter for the kind of workload
mixed-precision tensor cores target.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over the fixed 64-element fixture, requiring `max_abs_err <=
0.02` against the full-fp32 `numpy.dot(a, b)` — loose enough to accept
the reference's expected ~`0.003` quantization error, tight enough that
skipping the quantization (using the raw fp32 values directly) is also
completely acceptable numerically, but forgetting to combine all 32
lanes' partial sums (returning just one lane's 2-element contribution
instead of the full 64-element reduction) is off by orders of magnitude
and fails immediately. The empty starter leaves `out[0]` at its `-999.0`
sentinel.
