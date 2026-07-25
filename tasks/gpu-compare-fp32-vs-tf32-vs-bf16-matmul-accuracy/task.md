## Context

Tensor cores support several input formats that trade mantissa
precision for throughput, all normally accumulated at higher precision:
`fp32` (23 explicit mantissa bits), NVIDIA's `tf32` (10, same exponent
range as fp32), and `bf16` (7, also same exponent range as fp32). Fewer
mantissa bits means each *input* is rounded to a coarser grid before the
multiply-accumulate ever happens — the accumulator can be as precise as
you like, it can't recover information the inputs already lost.

Rounding a value to $m$ explicit mantissa bits, without any bitwise
tricks, is just rounding to the nearest multiple of a power-of-two step
sized to that value's own exponent: for $v \ne 0$ with binary exponent
$e = \lfloor \log_2 |v| \rfloor$, the rounding step is
$s = 2^{e - m}$, and the rounded value is
$\mathrm{sign}(v) \cdot \lfloor |v|/s + 0.5 \rfloor \cdot s$.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void quantized_matmul(const float* A, const float* B, float* C,
                                  int N, float mantissa_bits);
```

One thread per output element (`idx = blockIdx.x * blockDim.x +
threadIdx.x`, `row = idx / N`, `col = idx % N`). For that `(row, col)`,
compute

$$
C_{\text{row,col}} = \sum_{k=0}^{N-1} \mathrm{round}(A_{\text{row},k}, m) \cdot \mathrm{round}(B_{k,\text{col}}, m)
$$

where $\mathrm{round}(v, m)$ is the mantissa-rounding formula above and
$m$ is `mantissa_bits`. Round **both** operands of every product before
multiplying — accumulate the products at ordinary precision.

## Example

$v = 3.0$, $m = 2$: $e = \lfloor \log_2 3 \rfloor = 1$, $s = 2^{1-2} =
0.5$. $\lfloor 3.0/0.5 + 0.5 \rfloor \times 0.5 = \lfloor 6.5 \rfloor
\times 0.5 = 6 \times 0.5 = 3.0$ (unchanged — 3.0 needs only 2 explicit
mantissa bits to represent exactly: $1.1_2 \times 2^1$). $v = 3.3$,
$m=2$: same $e, s$; $\lfloor 3.3/0.5+0.5\rfloor \times 0.5 = \lfloor
7.1\rfloor \times 0.5 = 3.5$ — rounded to the nearest quarter-step.

## What the gate checks

The grader parses `solve.cu` and launches `quantized_matmul` three
times against the same fixed $8\times8$ matrices (seeded, values in
$[-2,2]$), once each with `mantissa_bits` set to `23.0` (fp32), `10.0`
(tf32), and `7.0` (bf16). Each run's output is compared against an
exact float64 matmul of the *un-rounded* matrices (`A @ B` in numpy).
It requires all of:

$$
\mathrm{fp32\_rel\_err} \le 10^{-5}, \quad
5\times10^{-5} \le \mathrm{tf32\_rel\_err} \le 2\times10^{-3}, \quad
5\times10^{-4} \le \mathrm{bf16\_rel\_err} \le 10^{-2}
$$

The lower bounds matter as much as the upper ones: a kernel that simply
ignores `mantissa_bits` and computes an ordinary, un-rounded matmul
would trivially satisfy every upper bound (skipping rounding can only
make the result *more* accurate) but reports a relative error around
$10^{-16}$ for every precision — far below the lower bounds — so it's
caught. On this fixture, the reference measures **fp32: 2.84e-08**,
**tf32: 2.36e-04**, **bf16: 2.05e-03**: three cleanly separated bands,
each about an order of magnitude worse than the last, purely from
losing mantissa bits before the multiply.
