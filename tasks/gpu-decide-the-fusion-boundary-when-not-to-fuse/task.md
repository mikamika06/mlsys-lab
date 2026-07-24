## Context

Kernel fusion removes a global-memory boundary between adjacent operations. If an
intermediate tensor is materialized, the producer stores it and the consumer loads
it. For an edge with tensor size $s_i$, the cut cost is

$$C_{\mathrm{cut}}(i) = 2s_i.$$

Fusion can also be a bad choice. If the producer value is reused by $u_i$
consumers, fusing it into one downstream region can force recomputation for the
other consumers. With recompute cost $r_i$, the fusion cost is

$$C_{\mathrm{fuse}}(i) = r_i(u_i - 1).$$

The local traffic-minimizing decision for edge $i$ is to create a fusion boundary
when

$$C_{\mathrm{cut}}(i) \le C_{\mathrm{fuse}}(i),$$

and to fuse across the edge otherwise. The output is a cut vector: $1$ means
"do not fuse across this edge", and $0$ means "fuse across this edge".

## Task

Implement the SIMT kernel:

```python
def fusion_boundary_kernel(t, graph_base, out_base, num_cases, max_edges):
    ...
```

The Arena software GPU calls your function once per simulated thread. Use the
GPU thread object only:

```python
t.gid
t.threadIdx
t.blockIdx
t.blockDim
t.gload(index)
t.gstore(index, value)
t.sload(index)
t.sstore(index, value)
t.alu(n)
```

The flattened graph data begins at `graph_base`. For case `c` and edge `e`, the
three integers are stored consecutively:

```python
offset = graph_base + (c * max_edges + e) * 3
s_i = gmem[offset + 0]
u_i = gmem[offset + 1]
r_i = gmem[offset + 2]
```

Write the decision for that edge to:

```python
out_base + c * max_edges + e
```

A value of `1` means a boundary is inserted, and a value of `0` means the two
operations are fused. Threads whose global id is outside the edge range should do
nothing.

## Example

For a chain with three candidate edges, suppose the stored triples are:

```python
[(16, 1, 99), (32, 4, 30), (20, 3, 7)]
```

The decisions are:

```python
edge 0: cut cost = 32, fuse cost = 0   -> fuse, write 0
edge 1: cut cost = 64, fuse cost = 90  -> cut,  write 1
edge 2: cut cost = 40, fuse cost = 14  -> fuse, write 0
```

So the output cut vector is:

```python
[0, 1, 0]
```

## What the gate checks

The grader constructs deterministic graph batches in global memory, launches your
kernel with the Arena software GPU simulator, and reads back the cut vector from
simulated global memory. It computes the reference decision from

$$C_{\mathrm{cut}}(i) = 2s_i$$

and

$$C_{\mathrm{fuse}}(i) = r_i(u_i - 1).$$

The `exact_match` gate passes only if every simulated global-memory output entry
matches the reference partition exactly.
