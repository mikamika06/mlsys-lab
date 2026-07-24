## Context

`add`, `multiply`, and `ReLU` are common building blocks chained together in
neural-network layers. For vectors $a, b, c \in \mathbb{R}^n$, the fused
computation is

$$
\mathrm{out}_i = \max\bigl(0,\ (a_i + b_i)\cdot c_i\bigr).
$$

A naive GPU implementation launches three separate kernels — one for the
add, one for the multiply, one for the ReLU — writing each stage's result
to global memory and reading it back in the next kernel. That's 2 extra
global writes and 2 extra global reads per element beyond the unavoidable
3 input reads and 1 output write.

**Kernel fusion** does all three steps inside a *single* kernel, keeping
every intermediate value in a register (a local scalar) instead of writing
it back to memory. Each thread then touches global memory exactly 4 times
total: one read of `a[i]`, one of `b[i]`, one of `c[i]`, and one write of
`out[i]` — regardless of how many arithmetic steps happen in between.

## Task

Implement the CUDA-C kernel

```cpp
__global__ void fuse(float* out, const float* a, const float* b, const float* c, int n);
```

for `i = blockIdx.x * blockDim.x + threadIdx.x`, guarded by `i < n`:

1. Read `a[i]`, `b[i]`, `c[i]` — each exactly once.
2. Compute `max(0, (a[i] + b[i]) * c[i])` using local scalar variables for
   every intermediate.
3. Write the result to `out[i]` exactly once.

Do **not** write an intermediate result to `out[i]` (or any other global
array) and then read it back for the next step — that defeats the whole
point of fusing the operations into one kernel.

## Example

For `a = [1, 2, 3]`, `b = [4, -5, 6]`, `c = [0, 2, -1]`:

```
(a+b)      = [5, -3, 9]
(a+b)*c    = [0, -6, -9]
max(0, .)  = [0, 0, 0]
```

## What the gate checks

*Correctness.* The grader compares `out` against a NumPy reference:

$$
\mathrm{max\_abs\_err} = \max_i \bigl|\widehat{\mathrm{out}}_i - \mathrm{out}_i\bigr| \le 10^{-9}
$$

*Fusion.* The simulator counts global-memory `transactions` (coalesced
128-byte segments touched, summed over every access the warp makes). A
genuinely fused kernel — 3 reads + 1 write per thread, nothing more —
measures **32** transactions on this fixture; a kernel that computes each
step by writing to `out[i]` and reading it back for the next step measures
**64**. The gate requires

$$
\mathrm{transactions} \le 40
$$

which a correctly-fused kernel clears comfortably and a step-by-step,
memory-round-tripping one does not, even though both can compute the
mathematically correct answer.
