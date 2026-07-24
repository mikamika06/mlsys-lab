## Context

Pointer chasing — following `next` links one node at a time — has zero
memory-level parallelism (MLP): the address of node `i+1` is the *value*
read from node `i`, so the CPU cannot issue that next load until the
current one has actually returned. Even a core that can juggle 8
outstanding cache misses can only ever have 1 in flight while chasing a
list.

Once the visiting order is known, though, re-reading the same data by
*index* is a completely different access pattern: every address is
computable from the index array alone, with no dependency on any other
load's result. That is exactly the access pattern the CPU's
outstanding-miss buffer was built to exploit.

## Task

Implement both:

```cpp
void pointer_chase_traversal(const int* next_idx, int head, int n, int* order_out);
void gather_by_index(const double* values, const int* order, int n, double* out);
```

`pointer_chase_traversal`: starting at `head`, follow `next_idx[]`
(`next_idx[i]` is node `i`'s successor, `-1` past the end) and write the
visited node indices into `order_out[0..n)` in visiting order. Call
`report_load(false)` once per step — see `sol.hpp` for exactly what the
harness does with it.

`gather_by_index`: for each `i`, set `out[i] = values[order[i]]`. Call
`report_load(true)` once per gather.

## Example

For a 50-node list, `pointer_chase_traversal` reports 50 dependent loads
that all extend one serial chain — the harness measures a critical path
of `50`. `gather_by_index`'s 50 independent loads round-robin across the
harness's 8-slot outstanding-load window, so its critical path comes out
at `ceil(50/8) = 7` — the whole benefit of having already materialized
the traversal order as a flat, indexable array.

## What the gate checks

`exact_match`: the driver prints both critical paths, the full 50-entry
traversal order, and the full 50-entry gathered-values array for one
fixed 50-node list. The traversal order must be byte-exact (any
off-by-one or wrong-successor bug shows up immediately in the printed
list); reporting the wrong load kind, or in the wrong function, changes
one of the two critical-path numbers. An empty starter leaves the
driver's `-1` / `-999.0` sentinels in place and fails outright.
