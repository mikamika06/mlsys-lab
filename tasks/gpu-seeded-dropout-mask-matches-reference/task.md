## Context

Dropout needs a per-element random decision — keep or drop — but a GPU
kernel has no shared, stateful RNG the way a CPU loop does: thousands of
threads run in whatever order the hardware schedules them, so a random
generator that depends on "the previous call" wouldn't even give the same
mask twice on the same seed. Real dropout kernels use a **counter-based**
scheme instead: each element's decision is a pure function of `(seed,
index)` alone, with no sequential dependency on any other element. Any
thread, launched with any grid/block shape, computes the exact same
decision for element `i` as any other launch would — that's what makes the
mask reproducible.

This CUDA-C subset has no RNG builtin and no bitwise operators either (so
no xorshift-style hash), so the hash here is built from pure
multiplication and modulo — a large fixed multiplier, a large prime
modulus, dividing down to a pseudo-uniform value in `[0, 1)`:

$$
h = (\text{seed} + i \times 2654435761) \bmod 1000000007
\qquad
u = h / 1000000007
$$

$$
\text{mask}[i] = \begin{cases} 0 & u < p \text{ (drop)} \\ 1 & u \ge p
\text{ (keep)} \end{cases}
$$

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void dropout_mask(float* mask, int n, int seed, float p);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x` in `[0, n)`, compute `h` and
`u` exactly as above, then write `mask[i] = 0.0f` if `u < p`, else
`mask[i] = 1.0f`.

## Example

With `seed = 12345`, `p = 0.3`: element `0` gives
`h = 12345`, `u \approx 0.0000123` — below `p`, so `mask[0] = 0.0` (dropped).
Element `1` gives `h = 654448092`, `u \approx 0.6544` — above `p`, so
`mask[1] = 1.0` (kept). Over the full 200-element fixture, almost exactly
`30%` of elements come out `0.0`, matching `p = 0.3`.

## What the gate checks

`check.py` recomputes the same hash independently in Python for
`n = 200`, parses `solve.cu`, and runs `dropout_mask` on the software GPU
(`arena.cuda_sim.GPU`) with a 7-block, 32-thread launch (224 threads > 200,
exercising the tail guard too). It requires `max_abs_err == 0.0` against
that independently computed mask — every element's decision has to match
bit-for-bit, not just the overall drop rate; a hash that swaps the roles
of `seed` and `i*2654435761`, or uses `<=` instead of `<` against `p`,
produces a *statistically* similar mask (same drop rate) but disagrees
with the reference on individual elements.
