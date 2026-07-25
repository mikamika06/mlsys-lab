## Context

**Memory-level parallelism (MLP)** is the time-averaged number of cache
misses a CPU keeps in flight at once. A core with out-of-order execution
can overlap multiple outstanding misses — while one is waiting on DRAM,
another can already be in flight — so high-MLP access patterns hide memory
latency far better than low-MLP ones, even though every individual miss
still takes the same number of cycles to complete.

Model a memory system as `window` outstanding-miss slots, each miss taking
`latency` cycles. Processing miss $i$:

- if the pattern is **chained** (its address depends on the *data* returned
  by the previous miss — pointer chasing), miss $i$ cannot even be issued
  until miss $i-1$ has fully completed, no matter how large `window` is;
- otherwise the address is known up front (computed from a loop counter,
  never from previously-loaded data), so miss $i$ only needs a free slot in
  the `window`-wide buffer, and independent misses overlap freely.

$$
\mathrm{cycles} = \text{completion time of the last miss}, \qquad
\mathrm{mlp} \times 1000 = \left\lfloor \frac{n_{\text{misses}} \cdot \mathrm{latency} \cdot 1000}{\mathrm{cycles}} \right\rfloor
$$

`mlp` is the fraction of total miss-service-time that fits into the elapsed
wall-clock time — exactly 1 when everything is fully serial, and up to
`window` when everything overlaps perfectly.

## Task

Implement

```cpp
struct MlpResult { long long cycles; long long mlp_x1000; };
MlpResult simulate_mlp(int n_misses, int window, int latency, bool chained);
```

Simulate `n_misses` misses in order, using `window` outstanding-miss slots
that each take `latency` cycles once occupied:

1. Miss $i$'s **ready time** is the completion time of miss $i-1$ if
   `chained`, else $0$.
2. Assign it to whichever slot frees up **earliest**. Its **issue time** is
   $\max(\text{ready}, \text{that slot's free time})$; it **completes**
   `latency` cycles after issuing, and the slot's free time becomes that
   completion time.

Return `{cycles, mlp_x1000}` as defined above (integer division, truncating).

## Example

`simulate_mlp(4, 8, 100, false)`: 4 independent misses, 8 free slots — all
4 issue at time 0 and complete at time 100. `cycles = 100`,
`mlp_x1000 = 4*100*1000/100 = 4000`.

`simulate_mlp(64, 8, 100, true)`: chained, so every miss waits for the
previous one — 64 serial hops of 100 cycles each. `cycles = 6400`,
`mlp_x1000 = 64*100*1000/6400 = 1000` (MLP exactly 1, as expected for a
fully serial dependency chain).

## What the gate checks

`main.cpp` reduces four archetypal traversal patterns to
`(n_misses, window, latency, chained)` parameters (documented in the file):
**pointer_chase** (64 misses, chained), **sequential** (a 64-element
unit-stride walk touches only 4 distinct 64-byte lines, independent,
window 8), **strided** (64 distinct-line misses, independent, window 8),
and **scatter_gather** (64 independent misses expressed as concurrent
streams, window 16). It calls your `simulate_mlp` on each, prints every
`cycles`/`mlp_x1000` pair, then sorts the four pattern indices ascending by
`mlp_x1000` and prints that order. The grader compiles your `.cpp` with the
real local `clang++`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed number (all cycles, all mlp\_x1000, and the final order) matches the reference}
$$

The correct ranking comes out to `pointer_chase < sequential < strided <
scatter_gather` (mlp_x1000 = 1000, 4000, 8000, 16000) — but that ranking is
only checked indirectly: a `simulate_mlp` that gets even one raw
`cycles`/`mlp_x1000` number wrong (e.g. by ignoring `chained`, or always
returning a fixed value) fails immediately, even if it happens to guess the
right final order.
