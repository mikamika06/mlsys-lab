## Context

Shared memory is split into 32 banks (one 4-byte word per bank, wrapping
every 32 words). A warp's 32 lanes can all be served in a single cycle
only if they touch 32 *different* banks; if several lanes land on the
same bank, those accesses serialize.

Store an `N x N` tile row-major (`phys = row*32 + col`) and everything
looks fine for a **row** access (fixed `row`, `col = 0..31`): consecutive
addresses, 32 different banks, no conflict. But a **column** access
(fixed `col`, `row = 0..31`) touches addresses `col, 32+col, 64+col,
...` — every one of them has the *same* `phys % 32 == col`. All 32
lanes fight over one bank: a full 32-way conflict, on the exact same
data a row access reads for free.

A **swizzle** re-maps `(row, col)` to a different physical slot so that
*both* row and column accesses stay conflict-free, without touching how
much memory is used or changing which values live where in the tile
logically. The classic trick shifts each row's column by an amount
that depends on the row: $\mathrm{phys}(row, col) = row \times 32 +
((row + col) \bmod 32)$. For a fixed row, this is just `col` relabeled
by a constant shift — still all 32 residues, still conflict-free. For a
fixed *column*, as `row` sweeps `0..31`, `(row + col) \bmod 32` sweeps
all 32 residues too (adding a changing constant mod 32 to a fixed value
is a bijection) — so a column access is now conflict-free as well.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void swizzle_roundtrip(const float* in, float* out, int target_col);
```

One warp (32 threads, `row = threadIdx.x`). Using
`phys(row, col) = row*32 + (row+col) % 32`:

1. For `col` from `0` to `31`: `tile[phys(row, col)] = in[row*32 + col]`
   (store the whole row into shared memory through the swizzle).
2. `__syncthreads()`.
3. `out[row] = tile[phys(row, target_col)]` (read back column
   `target_col` through the same swizzle).

`tile` is `__shared__ float tile[1024]` (a `32 x 32` tile).

## Example

`row=3, col=7`: `phys = 3*32 + (3+7)%32 = 96 + 10 = 106`. For the same
column `7` but `row=29`: `phys = 29*32 + (29+7)%32 = 928 + 4 = 932` —
banks `106 % 32 = 10` and `932 % 32 = 4`: different banks, even though
both lanes are reading "column 7".

## What the gate checks

The grader launches `swizzle_roundtrip` with a fixed random `32 x 32`
tile and `target_col = 7`, then checks the retrieved column against
`A[:, 7]` directly, and reads `smem_waves` straight from the simulator's
own bank-conflict model (not a hand-checked invariant). It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9} \quad\text{and}\quad \mathrm{smem\_waves} \le 40
$$

Correctness alone doesn't imply a good swizzle — the *identity* mapping
`phys = row*32 + col` is also a perfectly valid bijection (every slot
still gets exactly one value, so `max_abs_err = 0` too) but measures
**1056** shared-memory waves: every one of the 32 per-row store steps
*and* the final column read all suffer full 32-way conflicts. The
diagonal swizzle above measures **33** — 32 conflict-free store steps
plus one conflict-free read, a 32x reduction from nothing but which
physical slot each value lands in.
