## Context

The **roofline model** characterises a machine by two ceilings:
peak floating-point throughput $P$ (in FLOP/s) and peak memory bandwidth $B$ (in
byte/s).  The **ridge point** (or balance point) is the arithmetic intensity at
which the two ceilings meet:

$$I^{*} = \frac{P}{B}$$

For a kernel whose operational intensity $I$ (FLOP/byte) satisfies $I < I^{*}$
the kernel is **memory-bound**; when $I > I^{*}$ it is **compute-bound**.  A
dot-product of two $n$-element `float64` arrays performs $2n$ FLOPs while
touching $16n$ bytes (two reads of 8 bytes each), giving an arithmetic intensity
$I = 2n/(16n) = 1/8$ FLOP/byte — typically well below the ridge point, so it is
memory-bound on most machines.

Memory is delivered in 64-byte **cache lines**.  A naïve access pattern that
jumps by a column stride re-fetches the same line repeatedly.  Processing data
**sequentially in chunks** that fit in L1 keeps each line in cache and reuses it
across the inner loop, producing minimal cache misses.

## Task

Implement two functions:

1. **`ridge_point(peak_flops, peak_bw)`** — given peak FLOP/s ($P$) and peak
   bandwidth in byte/s ($B$), return the ridge point $I^{*} = P / B$ (a `float`).

2. **`dot_trace(n, l1_bytes, line_bytes)`** — generate the byte-address access
   trace (a `list[int]`) for a dot-product kernel over two `float64` arrays of
   length $n$.  Addresses start at 0 (array `a`) and $8n$ (array `b`).  Process
   elements in chunks that fit inside L1 so the trace is cache-friendly.  Return
   the trace as a plain list of integer byte addresses in access order.

## Example

```python
I = ridge_point(1e12, 50e9)
# 1e12 / 50e9 = 20.0  FLOP / byte

trace = dot_trace(4, 256, 64)
# Process in chunks of 16 elements (256 / 16 = 16 fits in L1)
# trace starts with 0, 8, 16, ..., 120, 512, 520, ...
```

## What the gate checks

1. **`rel_err`** — the grader computes the reference ridge point
   $I^{*}_{\text{ref}} = P_{\text{ref}}/B_{\text{ref}}$ and checks
   $|I^{*}_{\text{yours}} - I^{*}_{\text{ref}}| / |I^{*}_{\text{ref}}| \le
   10^{-12}$.
2. **`covers_all`** — the grader converts each byte address in your trace to an
   element index (`addr // 8`) and verifies every element of both arrays appears
   ($\mathrm{covers\_all}=1$).
3. **`misses`** — the grader feeds your trace to a deterministic LRU cache
   simulator (L1 size = `l1_bytes`, line = `line_bytes`, 8-way, 64 sets) and
   counts misses.  A sequential, chunked access pattern yields $\le 512$ misses
   for the pinned parameters ($n=2048$, $\text{L1}=16384$, $\text{line}=64$).
