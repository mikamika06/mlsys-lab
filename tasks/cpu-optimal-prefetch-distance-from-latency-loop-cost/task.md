## Context

A software prefetch instruction starts a memory transfer before the data is
needed. How far ahead to issue it depends on two numbers: the memory
latency $L$ (cycles until the data arrives) and the cost $C$ of one loop
iteration's work (cycles between successive data consumptions). If the
prefetch for iteration $i$'s data is issued $d$ iterations early, it has
$d \cdot C$ cycles to land before iteration $i$ consumes it. For the
latency to be fully hidden, that time budget must cover $L$:

$$d \cdot C \ge L \quad\Longrightarrow\quad d = \left\lceil \frac{L}{C} \right\rceil .$$

A distance smaller than this leaves the processor waiting on every
iteration once steady state is reached — the loop becomes memory-bound no
matter how cheap the compute is. A distance at or above the derived value
hides the latency completely after a fixed, one-time startup cost.

## Task

Implement, in `solve.cpp`:

```cpp
int prefetch_distance(int mem_latency, int loop_body_cycles);
int count_stalls(int n, int mem_latency, int loop_body_cycles, int distance);
```

`prefetch_distance` returns $d = \lceil L / C \rceil$, the smallest integer
distance with `distance * loop_body_cycles >= mem_latency`.

`count_stalls` simulates a straight-line loop of `n` iterations against a
fixed-cost timeline: iteration `i` consumes its data at cycle
`i * loop_body_cycles`. The prefetch supplying iteration `i`'s data is
issued `distance` iterations earlier — at iteration
`max(i - distance, 0)` — and lands `mem_latency` cycles after being
issued. Iteration `i` **stalls** if its data has not landed by the time
it is consumed. Return the total number of stalling iterations over
`i in [0, n)`.

## Example

For $L = 120$, $C = 30$: $d = \lceil 120/30 \rceil = 4$. Running
`count_stalls(1000, 120, 30, 4)`: for `i >= 4` the prefetch issued at
`i - 4` lands exactly at cycle `(i-4)*30 + 120 == i*30` — no stall. Only
the first 4 iterations (whose prefetch is clamped to iteration 0) stall,
waiting out the initial pipeline fill:

```
prefetch_distance(120, 30) == 4
prefetch_distance(121, 30) == 5      // one cycle over the boundary rounds up
count_stalls(1000, 120, 30, 4) == 4  // fixed startup cost only
count_stalls(1000, 120, 30, 1) == 1000   // every iteration stalls: too shallow
```

With `distance = 1` (undersized), every iteration's prefetch is issued
only one iteration ahead, landing at `(i-1)*30 + 120 == i*30 + 90` —
always 90 cycles late relative to its consumption — so all 1000
iterations stall: the loop is memory-bound for its entire run instead of
paying a fixed, amortized startup cost.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `exact_match == 1` against the same driver linked
with `ref.cpp`: the derived distances for three `(L, C)` pairs and the
stall counts from both the derived distance and a fixed too-small
distance of 1 (over 1000 iterations) must all match exactly. The starter
returns `0` from both functions, so it prints the wrong distances and
`stalls_optimal=0` / `stalls_naive=0` instead of `4` / `1000` — getting
the arithmetic formula wrong, or getting the stall timeline's clamping
and comparison wrong, both fail the gate.
