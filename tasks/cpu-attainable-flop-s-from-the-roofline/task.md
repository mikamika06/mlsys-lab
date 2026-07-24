## Context

The **roofline model** answers: given a kernel's arithmetic intensity
(FLOPs performed per byte moved from DRAM), what throughput can it actually
attain on a given machine? Two ceilings apply simultaneously:

- The machine's **peak compute rate** ($P$, FLOP/s) — you can never exceed
  raw arithmetic throughput, no matter how much data reuse you have.
- The machine's **peak memory bandwidth** ($B$, bytes/s) times the kernel's
  **arithmetic intensity** ($I$, FLOPs/byte) — how many FLOPs the memory
  system can *feed* per second.

Whichever ceiling is lower is what the kernel actually attains:

$$
\text{attainable} = \min(P,\ I \times B)
$$

A kernel with low arithmetic intensity is **memory-bound** (the $I \times B$
term is the binding constraint); a kernel with high arithmetic intensity is
**compute-bound** (capped at $P$ instead).

## Task

Implement `attainable_flops(peak_flops, peak_bandwidth, arithmetic_intensity)`
(declared in `sol.hpp`): compute `arithmetic_intensity * peak_bandwidth`,
then return whichever of that or `peak_flops` is smaller.

## Example

```cpp
attainable_flops(200.0, 50.0, 10.0)  // bandwidth term 500 > 200 -> compute-bound: 200.0
attainable_flops(200.0, 50.0, 2.0)   // bandwidth term 100 < 200 -> memory-bound:  100.0
attainable_flops(200.0, 50.0, 4.0)   // bandwidth term 200 == 200 -> ridge point:  200.0
```

For the fixed driver's five scenarios, the correct run prints:

```
200.000000
100.000000
200.000000
40.000000
80.000000
```

A starter that unconditionally returns `0.0` prints five `0.000000` lines.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `max_abs_err <= 1e-9` against the same driver linked
with `ref.cpp`. Getting the direction of the `min` backwards, or computing
`peak_bandwidth / arithmetic_intensity` instead of the product, flips which
term wins on at least one of the five scenarios and fails the gate.
