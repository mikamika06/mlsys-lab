## Context

A KV-cache is typically stored `[layers, heads, seq_len, dim]`,
C-contiguous — `dim` is the fastest-varying axis in memory, then `seq`,
then `head`, then `layer`. A decode step needs to gather every layer's
and every head's key vector at the *current* sequence position: a slice
`K[:, :, t, :]`, one `dim`-length vector per `(layer, head)` pair,
scattered across the buffer at stride `seq_len * dim` from each other.

The *physical* layout is fixed — you don't get to change how the cache
was written. What you *do* control is which thread reads which
`(layer, head, d)` triple. Get that assignment backwards — say, thread
`id` cycling through `layer` fastest instead of `d` fastest — and the
gather is still 100% correct (every triple still gets read exactly once,
into exactly the right output slot), but consecutive threads now read
addresses `seq_len * dim` floats apart instead of adjacent ones: every
single lane lands in its own 128-byte segment instead of 32 lanes
sharing one.

## Task

Implement

```cuda
__global__ void decode_read(float* out, const float* kv,
                             int layers, int heads, int seq_len, int dim, int t);
```

`kv` is `[layers, heads, seq_len, dim]` C-contiguous
(`kv[((layer*heads+head)*seq_len+t)*dim + d]`). Design how consecutive
thread ids `tid = blockIdx.x*blockDim.x + threadIdx.x` map to
`(layer, head, d)` triples so that **`d` is the fastest-varying axis
across thread ids** — matching the fastest-varying axis in `kv`'s
physical layout — then:

```
addr    = ((layer*heads + head) * seq_len + t) * dim + d
out_idx = (layer*heads + head) * dim + d
out[out_idx] = kv[addr]
```

Every `(layer, head, d)` triple must be covered by exactly one thread —
the fixed `out_idx` formula is what makes the *values* land correctly
regardless of which thread computes which triple; only the *mapping* of
`tid` to a triple is your design choice, and that choice is what
determines coalescing.

## Example

`layers=4, heads=8, dim=32`: with `d` fastest (`d = tid % dim`,
`head = (tid/dim) % heads`, `layer = tid / (dim*heads)`), the 32 threads
of one warp share the same `(layer, head)` and cover `d = 0..31` — 32
consecutive floats, one 128-byte segment. With `layer` fastest instead
(`layer = tid % layers`, ...), a warp's 32 threads span 8 different
`head` values and jump `seq_len*dim` floats apart on nearly every step —
every lane opens its own segment.

## What the gate checks

`check.py` seeds a fixed random `[4, 8, 16, 32]` KV-cache, parses
`solve.cu`, and launches `decode_read` over `layers*heads*dim = 1024`
threads (`t=5`) on the software GPU. It compares `out` against a numpy
oracle (`kv[:, :, 5, :]`, computed from the same seeded array) and reads
the simulator's `transactions` count (128-byte segments touched, across
both the reads and the writes). It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6} \quad \text{and} \quad \mathrm{transactions} \le 70
$$

The reference gets both exactly right: `max_abs_err=0.0`,
`transactions=64` — one segment per warp for the read, one for the write,
32 warps each. A `layer`-fastest mapping is *equally correct*
(`max_abs_err=0.0` — every triple still lands in the right slot) but
scores `transactions=2048`, 32x worse, purely from which axis got
assigned to consecutive thread ids.
