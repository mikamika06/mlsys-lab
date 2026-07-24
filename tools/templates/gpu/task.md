## Context

On a GPU, threads run in **warps** of 32 lanes. When a warp reads global memory, the
hardware serves it in fixed 128-byte **transactions**. If the 32 lanes touch one
contiguous 128-byte segment the access is **coalesced** (1 transaction); scattered
addresses cost many transactions and are much slower.

## Task

Write the SIMT kernel `scale_kernel(t, N, a)` that computes $g[i] = a \cdot g[i]$ for
one element per thread, with **coalesced** access. Use `t.gid`, `t.gload(idx)`,
`t.gstore(idx, val)`, `t.alu(n)`.

## Example

```python
# thread t with t.gid == i does:
v = t.gload(i)        # read g[i]
t.gstore(i, a * v)    # write g[i]
```

## What the gate checks

The software GPU runs your kernel over the grid and checks correctness
($\mathrm{max\_abs\_err} \le 10^{-9}$) **and** that global access is coalesced
($\mathrm{transactions} \le 20$; a strided kernel emits far more).
