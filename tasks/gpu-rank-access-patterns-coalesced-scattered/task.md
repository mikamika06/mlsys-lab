## Context

The GPU serves global memory in fixed-size (128-byte) **transactions**. If
every lane of a 32-lane warp touches an address inside the *same* segment,
the whole warp is served by one transaction — a **coalesced** access. If the
warp's addresses spread across several segments, the hardware issues one
transaction per segment touched.

The key subtlety: coalescing depends on which **segments** a warp touches as
a *set* — not on which lane maps to which address within that set, and not
on the order lanes are numbered in. A warp whose 32 lanes touch 32
consecutive addresses is coalesced whether lane 0 gets the lowest address or
the highest one. Only *spreading out* — a stride between consecutive lanes'
addresses — actually costs extra transactions.

## Task

Implement three kernels, each scaling `n` elements of `g` by `a`, but at a
different address for thread `idx = blockIdx.x * blockDim.x + threadIdx.x`:

```cpp
__global__ void unit_stride(float* g, float a, int n);      // g[idx]
__global__ void reversed_stride(float* g, float a, int n);  // g[n - 1 - idx]
__global__ void stride4(float* g, float a, int n);          // g[idx * 4]
```

Guard every access with `idx < n`.

## Example

For `n = 4`: `unit_stride` touches `g[0], g[1], g[2], g[3]`;
`reversed_stride` touches the *same four* addresses, `g[3], g[2], g[1],
g[0]` — same set, reversed per-thread assignment; `stride4` touches `g[0],
g[4], g[8], g[12]` — four addresses spread four times as far apart.

## What the gate checks

Each kernel runs on its own fresh GPU. The grader compares each one's
output against a NumPy reference (`unit_stride` and `reversed_stride` are
expected to produce the *identical* final array, since they touch the same
elements):

$$
\mathrm{max\_abs\_err} \le 10^{-9}
$$

and reads the REAL measured transaction count for each:

$$
\mathrm{transactions\_unit} \le 8 \qquad
\mathrm{transactions\_reversed} \le 8 \qquad
\mathrm{transactions\_stride4} \ge 12
$$

`unit_stride` and `reversed_stride` measure identically low (both
coalesced); `stride4` measures roughly 4x higher. Implementing
`reversed_stride` as if it needed special "gather" handling (e.g. only
touching `g[idx]`) would fail correctness outright, since it wouldn't
actually reach `g[n-1-idx]` — there's no way to pass the correctness gate
here without genuinely reversing the addressing, and no way to pass the
`stride4` gate without genuinely striding it.
