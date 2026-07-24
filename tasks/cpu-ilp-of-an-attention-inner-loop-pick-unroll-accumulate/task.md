## Context

An attention score loop that reduces into one running sum,
`acc += a[i] * b[i]`, is a serial dependency chain: iteration `i` cannot
begin until iteration `i-1`'s add has retired, so an `n`-element reduction
takes `n` sequential steps no matter how many execution ports the CPU
has sitting idle. The standard fix — "unroll and accumulate" — spreads
the work across several *independent* running sums, so the CPU's
out-of-order engine can overlap them; only the longest individual chain
sits on the critical path.

## Task

Implement

```cpp
double dot_product_ilp(const double* a, const double* b, int n, int num_chains);
```

Compute $\sum_i a_i b_i$ using `num_chains` independent accumulators,
round-robin assigned: element `i` accumulates into chain `i % num_chains`.
For every element, call `report_op(i % num_chains)` exactly once (see
`sol.hpp` — this is how the harness measures which chain ends up
longest). After the loop, sum the `num_chains` partial accumulators
together and return the total.

## Example

With `n = 97` and `num_chains = 4`, chain `0` receives elements
`0, 4, 8, ..., 96` — 25 elements — while chains `1..3` receive 24 each.
The critical path is the length of the *longest* chain: `25`, versus
`97` for a single running sum — roughly a 4x reduction, matching
`num_chains`.

## What the gate checks

`exact_match`: the driver prints the dot product value and the modeled
critical path (the report_op-tagged chain with the most steps) for a
fixed 97-element input with `num_chains = 4`. Using a single accumulator
(or forgetting to call `report_op` with the right chain id) makes the
critical path come out as `97` (or `0`) instead of `25`, failing the
match even if the dot-product value itself is right; a starter returning
`0.0` fails outright.
