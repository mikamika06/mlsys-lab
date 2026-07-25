## Context

An embedding table gather — thread $i$ fetches the full $D$-dimensional
row $\text{idx}[i]$ — can be laid out two ways:

- **AoS** (row-major / vocab-major): row $v$'s $D$ values are one
  contiguous run, at $\text{emb}[v \cdot D + d]$.
- **SoA** (dimension-major): dimension $d$'s values for *every* row form
  one contiguous run instead, at $\text{emb}[d \cdot V + v]$.

A common gather kernel has each thread loop over its own row's $D$
dimensions sequentially: `for d in [0,D): sum += emb[...]`. At loop step
$d$, every thread in the warp performs its access in lockstep — so what
matters for coalescing is how thread $i$'s address at that *one* step
compares to thread $i{+}1$'s.

Under **SoA**, step $d$'s address is $d \cdot V + \text{idx}[i]$: as $i$
increases across the warp, only the small $\text{idx}[i]$ term changes —
consecutive threads land on consecutive addresses, coalescing into a
single 128-byte transaction. Under **AoS**, step $d$'s address is
$\text{idx}[i] \cdot D + d$: consecutive threads are $D$ elements
*apart*, so each one falls in its own separate 128-byte segment — a
transaction *per lane, per step*, instead of one for the whole warp.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void gather_soa(float* out, const float* emb, const int* idx, int D, int V);
```

`emb` is stored SoA: `emb[d*V + v]`. For thread `i = blockIdx.x*blockDim.x
+ threadIdx.x`, read `v = idx[i]`, sum `emb[d*V + v]` for `d` in `[0, D)`,
and write the sum to `out[i]`.

## Example

With `D=32`, `V=256`, a single 32-thread warp (`idx[i] = i`): the correct
SoA-indexed kernel measures `34` total transactions on the simulator (1
for the `idx` load, 32 — one per dimension step, each a single coalesced
128-byte transaction — for the `emb` reads, 1 for the `out` store). The
same computation with the AoS formula (`emb[v*D + d]`) instead measures
`1026` — one scattered transaction per lane, for every one of the 32
dimension steps — over 30x more traffic to move the exact same useful
bytes.

## What the gate checks

`max_abs_err <= 1e-6` (each `out[i]` must equal the true sum of row
`idx[i]`'s 32 dimensions) **and** `transactions <= 40` on the fixed
32-thread launch. Indexing with `v*D + d` instead of `d*V + v`, or
swapping the loop to iterate over rows instead of dimensions, still
computes the right sum (or fails outright) but blows the transaction
budget — this task is graded on the real simulated SIMT memory-access
pattern, not just the final numbers.
