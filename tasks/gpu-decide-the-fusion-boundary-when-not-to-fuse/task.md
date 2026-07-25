## Context

Kernel fusion removes a global-memory boundary between two adjacent
operations. If an intermediate tensor is instead materialized, the producer
stores it and the consumer loads it back. For an edge with tensor size
$s_i$, the cut cost (keep the boundary, pay for one store + one load) is

$$
C_{\mathrm{cut}}(i) = 2 s_i.
$$

Fusion is not always better: if the producer's value is reused by $u_i$
consumers, fusing it into a downstream region means only the first consumer
gets it "for free" — every other consumer forces the producer's work to be
recomputed. With per-recompute cost $r_i$, the fusion cost is

$$
C_{\mathrm{fuse}}(i) = r_i (u_i - 1).
$$

The traffic-minimizing decision for edge $i$ is to cut (insert a boundary,
i.e. materialize and don't fuse) when

$$
C_{\mathrm{cut}}(i) \le C_{\mathrm{fuse}}(i),
$$

and to fuse across the edge otherwise. Output `1` for "cut" and `0` for
"fuse".

## Task

Implement the CUDA kernel

```cuda
__global__ void fusion_boundary(const int* size, const int* reuse, const int* recompute, int* out, int n);
```

which, for every edge `i` in `[0, n)`, computes `cut_cost = 2 * size[i]`,
`fuse_cost = recompute[i] * (reuse[i] - 1)`, and writes `out[i] = 1` if
`cut_cost <= fuse_cost`, else `out[i] = 0`. Guard `i < n`.

## Example

For three edges with `(size, reuse, recompute)` triples
`(16, 1, 99), (32, 4, 30), (20, 3, 7)`:

```
edge 0: cut = 32,  fuse = 99*(1-1) = 0   -> fuse, out = 0
edge 1: cut = 64,  fuse = 30*(4-1) = 90  -> cut,  out = 1
edge 2: cut = 40,  fuse = 7*(3-1)  = 14  -> fuse, out = 0
```

So `out = [0, 1, 0]`.

## What the gate checks

`check.py` builds a deterministic batch of 144 edges (sizes, reuse counts,
and recompute costs derived from a fixed arithmetic mix, with a subset
pushed exactly onto the cut/fuse tie point), uploads `size`, `reuse`,
`recompute` into the software GPU's global memory, parses and launches
`solve.cu`'s `fusion_boundary` kernel, and reads the `out` cut-vector back.
It compares against a numpy oracle computed from the same
$C_{\mathrm{cut}} = 2s_i$, $C_{\mathrm{fuse}} = r_i(u_i-1)$ formulas. The
`exact_match` gate requires every one of the 144 entries to match. Dropping
the `- 1` (charging `recompute * reuse` instead of `recompute * (reuse -
1)`, i.e. forgetting the first consumer gets the producer's value for free)
overcounts the fusion cost on every single edge and flips the decision on
several of them — including every edge deliberately placed on the tie
point, which is exactly what this batch is built to catch.
