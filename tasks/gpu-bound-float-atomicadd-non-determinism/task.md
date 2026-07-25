## Context

Real GPU reductions built on `atomicAdd` are famously not bitwise
reproducible: which thread's update lands first, second, third is
decided by hardware scheduling, not the program. Since floating-point
addition is **not associative** — $(a+b)+c$ and $a+(b+c)$ can round to
different results — a different arrival order can produce a measurably
different final sum, even summing the exact same values. Avoiding
`atomicAdd` in favor of a deterministic **tree reduction** (halving the
active thread count each round, always combining the same shared-memory
slots) removes the *scheduling* non-determinism, but the same
non-associativity issue reappears the moment the *input order* changes
— which is exactly what varies, run to run, when values arrive from a
race.

This non-determinism isn't unbounded chaos, though. The classical
Wilkinson bound on summing $n$ floating-point numbers with machine
epsilon $u$ says the computed sum can't be farther from the true sum
than $(n-1) \, u \sum_i |x_i|$ — and since two *different* orderings can
each independently deviate that far, in opposite directions, the
**spread** between any two orderings' results is bounded by

$$
\text{spread} \le 2(n-1)\,u\sum_i |x_i|
$$

## Task

Implement, in `solve.cu`:

```cuda
__global__ void block_reduce_sum(const float* x, float* out, int n);
```

A standard block-level binary tree reduction over `n = 64` values, one
per thread: load `x[tid]` into `__shared__ float s[64]`, `__syncthreads()`,
then with `stride` starting at `blockDim.x / 2` and halving each round:
while `stride > 0`, every thread with `tid < stride` does
`s[tid] = s[tid] + s[tid + stride]`, then `__syncthreads()`, then
`stride = stride / 2`. Once `stride` reaches `0`, thread `0` writes
`out[0] = s[0]`.

## Example

For `n = 4`, values `[a, b, c, d]` at shared-memory slots `0..3`: round 1
(`stride=2`) computes `s[0]=a+c`, `s[1]=b+d`; round 2 (`stride=1`)
computes `s[0]=(a+c)+(b+d)`. Feed the same four *values* in through a
different slot assignment — say `[c, a, d, b]` — and round 1 instead
computes `s[0]=c+d`, `s[1]=a+b`, giving `s[0]=(c+d)+(a+b)` — the same
four addends, a different association, generally a different rounded
result.

## What the gate checks

The grader builds a fixed 64-value fixture (32 values around
$10^{16}$, 32 around $1$ — large enough magnitude spread that
reordering genuinely changes the rounded sum) and, for 40 different
random permutations of it, uploads each permutation and launches your
kernel once. It requires:

$$
\mathrm{sums\_match} = 1 \iff \text{every one of the 40 sums matches the reference implementation's sum for that same permutation}
$$

$$
\mathrm{within\_bound} = 1 \iff \max(\text{sums}) - \min(\text{sums}) \le 2(n-1)\,u\sum_i|x_i|
$$

with $u = 2^{-52}$ (the simulator's floating-point machine epsilon).
On this fixture, the reference's 40 orderings span a real, nonzero
**spread of 64.0** between the highest and lowest sum — genuine,
measured, bitwise non-reproducibility from nothing but addition order —
while the derived bound is **8952.8**: the observed chaos is real, but
nowhere near unbounded.
