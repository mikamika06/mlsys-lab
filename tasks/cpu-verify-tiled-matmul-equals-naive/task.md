## Context

Cache blocking (tiling) a matmul changes the ORDER in which the
`A[i][k]*B[k][j]` products get summed into each `C[i][j]` — it groups
the reduction dimension into chunks of `tile` instead of running straight
through `k = 0..N-1` — but every product still gets computed and summed
exactly once. When `N` is not a multiple of `tile`, the boundary blocks
are simply narrower; they must not be skipped or double-counted.

## Task

Implement

```cpp
void tiled_matmul(const float* A, const float* B, float* C, int N, int tile);
```

Compute `C = A * B` for `N x N` row-major matrices using the blocked
algorithm: loop over block starts `(ii, jj, kk)` in steps of `tile`
(clamping each block's end to `N`), and within each block accumulate the
ordinary triple-loop product into `C[i][j]`. Zero `C` before
accumulating. `tile` does not have to evenly divide `N`.

## Example

With `N = 7` and `tile = 3`, the block starts are `0, 3, 6` in each
dimension — the last block in every dimension is just 1 wide instead of
3. `C[6][6]` still needs contributions from all three `k`-blocks
(`k` in `0..2`, `3..5`, and just `6`) accumulated together, not only the
first two.

## What the gate checks

`max_abs_err` on the full 7x7 output matrix for one fixed pair of
inputs, against a naive (unblocked) reference. Dropping the partial
boundary blocks, double-counting a block, or not zeroing `C` first
produces visibly wrong entries, especially in the last row/column; a
starter that leaves `C` zeroed fails outright since the real product is
not all zero.
