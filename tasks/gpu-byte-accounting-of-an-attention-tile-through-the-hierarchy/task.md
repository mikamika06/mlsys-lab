## Context

FlashAttention's core trick is to never materialize the full attention
matrix; instead it processes the sequence in **tiles**, and each tile
step touches every level of the memory hierarchy for a different reason:

- **Global memory** (slow, off-chip): where $Q$, $K$, $V$ actually live.
  Every byte pulled from here costs a real, expensive transaction.
- **Shared memory** (fast, on-chip, per-block): a scratchpad the whole
  thread block can read from repeatedly once a tile has been loaded
  into it.
- **Registers** (fastest, per-thread): where the running dot-product
  accumulator lives while a thread reduces over the head dimension.

For one $Q_{\text{tile}} \in \mathbb{R}^{B_Q \times D}$,
$K_{\text{tile}} \in \mathbb{R}^{B_K \times D}$ score computation
$S = Q_{\text{tile}} K_{\text{tile}}^\top$, the **byte-honest** version
of the computation loads each of the $B_Q \cdot D$ elements of $Q$ and
each of the $B_K \cdot D$ elements of $K$ from global memory exactly
**once**, parks them in shared memory, and then computes all
$B_Q \times B_K$ dot products (each a reduction over $D$) by reading
shared memory and accumulating in a register -- global memory is never
touched again during the reduction. A version that instead re-reads
`Q[i][d]` and `K[j][d]` directly from global memory inside each of the
$B_Q \times B_K$ threads' reduction loops is computing the exact same
answer, but re-fetches every element of $Q$ from global memory $B_K$
times (once per key row it's compared against) and every element of $K$
$B_Q$ times -- $B_Q \cdot B_K \cdot D \cdot 2$ global loads instead of
$(B_Q + B_K) \cdot D$.

## Task

Implement:

```cuda
__global__ void qk_tile(float* out, const float* Q, const float* K, int BQ, int BK, int D);
```

`Q` is a flattened $B_Q \times D$ tile, `K` a flattened $B_K \times D$
tile, both row-major. Launch with one block of `BQ*BK` threads (this
task fixes `BQ = BK = D = 8`, so `64` threads):

1. **Cooperative load**: with `BQ*D == BK*D == BQ*BK == 64` at this
   task's fixed sizes, thread `threadIdx.x` loads exactly one `Q`
   element and one `K` element into two `__shared__ float[64]` tiles
   (`Qs[tid] = Q[tid]; Ks[tid] = K[tid];`).
2. `__syncthreads()`.
3. Each thread computes its own output element $(i, j) = (tid / BK,\ tid
   \bmod BK)$: reduce over `d` in `[0, D)`, reading only from the
   shared tiles (`Qs[i*D+d] * Ks[j*D+d]`), accumulating into a plain
   local `float` register, and write the final sum to
   `out[i * BK + j]`.

## Example

For `BQ = BK = D = 2`, `Q = [1, 2, 3, 4]` (rows `[1,2]`, `[3,4]`),
`K = [5, 6, 7, 8]` (rows `[5,6]`, `[7,8]`): `out[0] = 1*5 + 2*6 = 17`,
`out[1] = 1*7 + 2*8 = 23`, `out[2] = 3*5 + 4*6 = 39`,
`out[3] = 3*7 + 4*8 = 53`. Every one of `Q`'s and `K`'s 4 elements is
read from global memory exactly once regardless of how many output
elements depend on it.

## What the gate checks

`check.py` runs the kernel over a fixed `8x8` `Q` tile and `8x8` `K`
tile (`D = 8`), one block, `64` threads. It checks `max_abs_err <=
1e-6` against `numpy`'s `Q @ K.T`, `transactions <= 20`, and
`cycles <= 5000` from the simulator's memory-hierarchy model. The
reference measures `6` transactions and `2344` cycles. A kernel that
skips the shared-memory tile and reduces straight out of `Q`/`K` in
global memory is still numerically correct, but measures `50`
transactions and `10104` cycles -- both well over the gate -- because
every one of the `64` output threads re-fetches its own `2*D` operands
from global memory instead of sharing one cooperative load across the
whole block.
