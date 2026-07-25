## Context

A weight-only-quantized linear layer stores its weight matrix as int8
codes plus a per-row (per-output-channel) floating-point scale, instead
of full-precision floats. The true weight is recovered by
**dequantizing**: $w_{\text{fp}}[i,j] = w_{\text{int}}[i,j] \cdot
\text{scale}[i]$. A matvec against it is

$$
y_i = \sum_{j} w_{\text{int}}[i,j] \cdot \text{scale}[i] \cdot x_j
$$

There are two ways to structure the kernel that computes this. A
**dequant-fused** kernel reads each int8 code from global memory once,
multiplies it by that row's scale right there in a register, and
immediately uses the result in the accumulate -- the dequantized value
never exists anywhere but a register, for exactly as long as it's
needed. The alternative -- materializing a full dequantized copy of the
weight matrix in memory before the matvec even starts -- would spend
extra global-memory traffic writing out (and then reading back) values
that are only ever used once each. Fusing the cheap dequant multiply
into the consuming instruction avoids that entirely: on a GPU, a
register multiply is essentially free next to a global-memory round
trip.

The other classic saving in a matvec is **not** re-fetching the shared
activation vector $x$ from global memory once per output row: with $M$
threads (one per row) all needing every element of $x$, loading it into
`__shared__` memory once and letting every thread read the fast on-chip
copy avoids $M{-}1$ redundant global reads of every element of $x$.

## Task

Implement:

```cuda
__global__ void dequant_matvec(float* y, const float* w_int, const float* scale, const float* x, int M, int N);
```

`w_int` is a flattened $M \times N$ matrix of int8-coded weights
(stored as plain numbers), `scale` has one entry per row, `x` has $N$
elements, `y` has $M$. Launch with one block of `M` threads (this task
fixes `M = N = 8`):

1. Cooperatively load `x` into `__shared__ float xs[8]`: thread
   `threadIdx.x` sets `xs[threadIdx.x] = x[threadIdx.x]`.
2. `__syncthreads()`.
3. Thread `i = threadIdx.x` computes row `i`'s output: read
   `scale[i]` once, then for `j` in `[0, N)`, read `w_int[i*N+j]` from
   global memory, dequantize it in a register
   (`w_int[i*N+j] * scale[i]`), and accumulate
   `acc += dequantized * xs[j]` (reading `x` from the shared copy, not
   global memory again). Write `y[i] = acc`.

## Example

For `M = N = 2`, `w_int = [10, -5, 3, 20]` (rows `[10,-5]`, `[3,20]`),
`scale = [0.1, 0.5]`, `x = [2, 4]`:
row 0: `(10*0.1)*2 + (-5*0.1)*4 = 2.0 - 2.0 = 0.0`.
row 1: `(3*0.5)*2 + (20*0.5)*4 = 3.0 + 40.0 = 43.0`.
`y = [0.0, 43.0]`.

## What the gate checks

`check.py` runs the kernel over a fixed `8x8` int8-coded weight matrix
with random per-row scales and a random `x`. It checks `max_abs_err <=
1e-6` against `numpy`'s `(w_int * scale[:,None]) @ x`, `transactions <=
22`, and `cycles <= 4500` from the simulator's memory-hierarchy model.
The reference measures `19` transactions and `4020` cycles. Skipping the
dequantization entirely (using the raw int8 codes as if they were
already the true weights) is still fast but wildly wrong numerically
(error over `100`, since scales here are around `0.01`-`0.03`) and fails
`max_abs_err`. Re-reading `x` from global memory inside every row's
loop instead of caching it in `__shared__` memory is numerically
correct but measures `26` transactions and `5240` cycles -- over both
performance gates.
