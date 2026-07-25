## Context

A loop that loads a value and then immediately uses it has a built-in bubble:
the load takes `latency` cycles to come back, and the consuming instruction
cannot issue until it does. Written the obvious way — finish iteration `i`
completely (issue its load, wait, consume) before even *issuing* iteration
`i+1`'s load — every single iteration pays the full latency as a stall.

**Software pipelining** breaks that dependency between iterations: issue
iteration `i+1`'s load while iteration `i`'s is still in flight (they don't
depend on each other), so its latency ticks away *underneath* other useful
work instead of blocking anything. Only the very first load has no earlier
work to hide behind — that one-time delay before the pipeline "fills" is the
only latency the loop ever pays; every iteration after that has its load
result ready exactly when the machine is ready to consume it.

## Task

Implement, in `solve.cpp`:

```cpp
struct StallCounts { long long naive_stalls; long long pipelined_stalls; };
StallCounts modeled_stall_counts(long long n, long long latency);
```

Model a 2-issue-per-cycle machine (one load-issue slot, one compute/consume
slot, both usable in the same cycle) running `n` iterations of a
load-then-consume loop, each load taking `latency` cycles to return. Every
iteration needs exactly 2 useful issue-events (1 load + 1 consume), which on
a 2-wide machine is a hard floor of `n` cycles regardless of scheduling —
"stalls" are cycles beyond that floor:

- `naive_stalls = n * latency` — each iteration serializes fully before the
  next one's load is issued, so the full latency is paid `n` times.
- `pipelined_stalls = latency` (or `0` if `n == 0`) — the fill cost is paid
  exactly once, no matter how many iterations follow it.

## Example

The driver (`main.cpp`, fixed) runs 5 `(n, latency)` pairs:

```
single_iteration n=1 latency=4 naive_stalls=4 pipelined_stalls=4
ten_iterations n=10 latency=4 naive_stalls=40 pipelined_stalls=4
thousand_iterations n=1000 latency=6 naive_stalls=6000 pipelined_stalls=6
zero_latency n=5 latency=0 naive_stalls=0 pipelined_stalls=0
zero_iterations n=0 latency=8 naive_stalls=0 pipelined_stalls=0
```

`single_iteration` shows pipelining has nothing to hide behind with only one
iteration — `naive_stalls` and `pipelined_stalls` are identical. From there
the gap only widens: `thousand_iterations` pays the same constant 6-cycle
fill cost pipelined, whether it runs 1000 iterations or 10, while the naive
schedule's stall count scales linearly with `n`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Reporting
`pipelined_stalls = n * latency` too (i.e. not modeling any overlap at all)
matches `single_iteration` by coincidence but is off by orders of magnitude
on `ten_iterations` and `thousand_iterations`; forgetting the `n == 0`
special case leaves `pipelined_stalls = 8` on `zero_iterations` instead of
`0`.
