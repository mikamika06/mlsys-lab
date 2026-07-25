## Context

A real low-bit matvec kernel doesn't unpack its weights into a separate
buffer first — it unpacks, dequantizes, and multiply-accumulates all in
the same inner loop, one packed byte at a time. This task fuses three
pieces already seen separately: **int4 nibble packing** (two codes per
byte, `b = lo + hi*16`), **group-wise dequant scales** (`scale[m, k/G]`,
one scale per group of `G` consecutive columns, per row), and a plain
dot product.

Row `m`'s weight vector (`K` int4 codes) is stored as `K/2` packed bytes:
byte `p` holds columns `2p` (low nibble) and `2p+1` (high nibble). Column
`k`'s code is scaled by its row's group-`k/G` scale before being
multiplied against `x[k]`.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void dequant_matvec(float* y, const float* packed_w, const float* scale,
                                const float* x, int M, int K, int G);
```

For row `m = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `m < M`:
for `p` in `[0, K/2)`, unpack `b = packed_w[m*(K/2) + p]` into
`hi = floorf(b / 16.0f)`, `lo = b - hi*16.0f`, then accumulate
`lo * scale[m*(K/G) + (2p)/G] * x[2p]` and
`hi * scale[m*(K/G) + (2p+1)/G] * x[2p+1]` into `y[m]`.

## Example

`K=8, G=4`: columns `0..3` are group `0`, columns `4..7` are group `1`.
Packed byte `p=1` holds columns `2` and `3` (both group `0`); packed byte
`p=2` holds columns `4` and `5` (both group `1`) — the group boundary
falls exactly on a packed-byte boundary here, but the *unpacking* and the
*group lookup* are still two independent steps: getting one right and
the other wrong (e.g. using group `p/2` instead of `k/G`) still fails.

## What the gate checks

`max_abs_err <= 1e-6` on a fixed `M=4, K=8, G=4` case (random codes,
scales spread `0.1`-`2.0`, random `x`), against a numpy oracle that
independently unpacks, dequantizes per group, and dots. Swapping which
nibble is `lo` vs `hi`, indexing `scale` by the wrong row or group, or
dropping the `floorf` (leaving `hi` fractional and corrupting `lo` too),
all diverge from the reference dot products.
