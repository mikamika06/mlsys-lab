## Context

Global-memory accesses **coalesce** when the 32 lanes of a warp touch
addresses that fall within the same handful of 128-byte segments — the
hardware serves the whole warp with one (or a few) memory transactions
instead of 32 separate ones. A transpose is the textbook case where
this quietly breaks: `out[c][r] = in[r][c]` reads `in` with `c` varying
fastest across the warp (contiguous, coalesced) but writes `out`
indexed the *other* way, with `c` now the *outer* (row) index of the
output — so consecutive lanes write addresses `N` floats apart. The
read is free; the write pays for one transaction *per lane*.

The fix doesn't avoid the strided direction — transposition is
inherently strided on one side, there's no avoiding it entirely — it
just confines the stride to **shared memory**, which doesn't have this
segment-transaction cost the way global memory does. Stage the tile
through shared memory with a coalesced read and a coalesced write on
the global-memory side; let the strided part happen only when reading
back out of the (fast, on-chip) shared tile.

## Task

`solve.cu`'s `transpose` writes directly to the strided global address.
**Fix it** by staging through shared memory instead:

1. Load `in[row][col]` into `tile[row*N+col]` — same address pattern as
   the input, still coalesced.
2. `__syncthreads()`.
3. Write `out[row][col] = tile[col][row]` — the **global** write address
   (`row*N+col`) is now the same contiguous pattern as any normal
   row-major write; only the **shared-memory** read (`tile[col*N+row]`)
   carries the strided part, and shared memory doesn't pay a
   per-lane-segment transaction cost for that.

`row = idx / N`, `col = idx % N`, `idx = blockIdx.x*blockDim.x +
threadIdx.x`; `tile` is `__shared__ float tile[1024]` (the whole
`32 x 32` matrix fits in one block).

## Example

Thread `idx=5` with `N=32`: `row=0, col=5`. Buggy write:
`out[5*32+0] = out[160]`. Thread `idx=6`: `row=0, col=6`, writes
`out[192]` — 32 floats (128 bytes) away from the previous lane's write,
a brand new segment every single lane. Fixed write: thread `idx=5`
writes `out[0*32+5] = out[5]`; thread `idx=6` writes `out[6]` —
adjacent, same segment as the rest of the warp.

## What the gate checks

The grader launches `transpose` on a fixed `32 x 32` matrix, checks the
output against `A.T` (numpy), and reads the real transaction count from
the simulator's coalescing model. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9} \quad\text{and}\quad \mathrm{transactions} \le 100
$$

Both versions are numerically perfect — this is purely a traffic bug.
The direct-write version measures **1056** transactions (the coalesced
read stays cheap, but every one of the 1024 uncoalesced writes costs
its own segment); staging through shared memory measures **64** — over
16x fewer, from writing the exact same 1024 values in a different
address order.
