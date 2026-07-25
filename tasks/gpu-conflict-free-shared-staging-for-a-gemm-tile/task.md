## Context

Shared memory is split into 32 equal-width banks; consecutive words land in
consecutive banks, wrapping around every 32 words. A warp (32 threads) can
service one access per lane in a single "wave" only if no two lanes hit
*different* words in the *same* bank — two lanes reading the exact same
word are a free broadcast, but two lanes reading different words that both
happen to land in bank 7 have to be serialized into separate waves.

Staging a GEMM tile through `__shared__` memory is the textbook case where
this bites. This kernel maps `tid -> (col = tid/16, row = tid%16)`, so a
32-lane warp spans **two full columns** — all 16 rows of each. During the
reduction it reads `As[row*lda + k]` for a fixed `k`, with `row` sweeping
every value 0..15 across the warp's lanes. If `As` is laid out with stride
`lda = n` (the "obvious" choice — no wasted space), those 16 different
`row` values land only 16 words apart, and $16 \bmod 32 = 16$: rows
$0, 2, 4, \ldots$ all fall in one bank, rows $1, 3, 5, \ldots$ all fall in
another. Padding the stride to $lda = n + 1 = 17$ fixes it — since
$\gcd(17, 32) = 1$, sixteen different `row` values now land in sixteen
*different* banks, no two lanes ever colliding.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void gemm_tile(float* C, const float* A, const float* B, int n);
```

One block, `n*n` threads (`n = 16`), `tid -> (col = tid/n, row = tid%n)`.
Stage `A` and `B` into `__shared__` arrays (`As`, `Bs`), choosing a padded
row stride for each — `n + 1` words, not `n` — so that no two lanes of a
warp ever read different words from the same bank during either the
staging step or the reduction. Then compute
`C[row][col] = sum_k A[row][k] * B[k][col]` by reading `As`/`Bs` (not `A`/`B`
again) inside the `k` loop, and write the result to `C[row*n + col]`.

## Example

The grader builds two random `16x16` matrices and computes their product
with an explicit Python triple loop, in the *same* left-to-right
accumulation order the kernel uses — so any numerically correct kernel
matches it **exactly**, `max_abs_err = 0.0`, no matter how its shared tiles
are laid out. Padding is a pure memory-layout decision; it cannot change
the arithmetic. It changes `smem_waves`:

```
unpadded (lda = ldb = n):       smem_waves = 1280
As padded only (lda = n+1):     smem_waves = 336
both padded (lda = ldb = n+1):  smem_waves = 288
```

Padding just `As` (the array whose reduction access varies `row` across the
warp) already recovers most of the benefit; padding `Bs` too closes the
rest of the gap — its own staging step has the same row-varies-across-warp
shape.

## What the gate checks

`check.py` builds the fixture, parses `solve.cu`, and runs `gemm_tile` on
the software GPU (`arena.cuda_sim.GPU`) with a 1-block, 256-thread launch.
It requires `max_abs_err == 0.0` (any correct summation order in this exact
accumulation scheme reproduces the reference bit-for-bit) **and**
`smem_waves <= 300`. The unpadded version above computes the exact right
answer — it passes the correctness gate outright — but its `smem_waves =
1280` is more than 4x the padded version's `288`, and fails the conflict
gate on its own.
