## Context

On a NUMA machine, a shared array's pages can be placed with two common
`numactl` policies:

- **`--membind`** ("bind"): every page lives on one chosen node. Accesses
  from that node are cheap (local DRAM); accesses from every other node
  cross the interconnect (remote, slower) -- but the *placement* itself
  is free to pick whichever single node is best.
- **`--interleave`**: pages are spread round-robin across every node.
  Regardless of which node issues an access, it lands on its "own"
  local slice of the data with probability $1/n$ and on a remote node's
  slice otherwise.

A pure latency argument alone can never favor interleave: binding to the
node with the largest share of accesses is always at least as good,
because a local access is never slower than a remote one. The real
reason interleave sometimes wins is **contention**: under bind, *every*
node's requests -- not just the bind node's own -- funnel through that
one node's single memory controller. The more nodes genuinely share the
array, the more that one controller queues under concurrent pressure.
Interleave sidesteps this entirely by spreading the load itself across
every controller.

This exercise models that queueing cost with the standard concentration
statistic, the inverse Herfindahl index, over each node's share
$f_k = \text{access\_count}_k / \text{total}$:

$$
\text{effective\_sharers} = \frac{1}{\sum_k f_k^2} \in [1, n]
$$

(1 when one node owns everything, $n$ when every node's share is equal).

## Task

Implement:

```cpp
const char* classify_workload(const Workload& w, double* bind_ns, double* interleave_ns);
```

`w.access_count[k]` (for `k` in `[0, w.num_nodes)`) is how many of a
workload's total accesses to one shared array originate from node `k`'s
threads. Using the pinned constants `LOCAL_NS`, `REMOTE_NS`,
`CONTENTION_COEF_NS` (declared in `sol.hpp`, defined in `main.cpp`):

1. Let `total = sum(access_count)` and `best = max(access_count)`, and
   `L = best / total`. This is the share the best possible single bind
   node would capture locally.
2. `base_bind_ns = L * LOCAL_NS + (1 - L) * REMOTE_NS`.
3. `effective_sharers = 1 / sum_k((access_count[k] / total)^2)`.
4. `*bind_ns = base_bind_ns + CONTENTION_COEF_NS * (effective_sharers - 1)`.
5. `*interleave_ns = LOCAL_NS / n + REMOTE_NS * (n - 1) / n` where
   `n = w.num_nodes` (no contention term: interleave's load is already
   spread evenly by construction).
6. Return `"interleave"` if `*interleave_ns < *bind_ns`, else `"bind"`.

## Example

With `LOCAL_NS = 90`, `REMOTE_NS = 190`, `CONTENTION_COEF_NS = 15`, and
`access_count = {9700, 100, 100, 100}` (`n = 4`):

- `total = 10000`, `best = 9700`, `L = 0.97`.
- `base_bind_ns = 0.97*90 + 0.03*190 = 93.0`.
- shares are `{0.97, 0.01, 0.01, 0.01}`, `sum(f^2) = 0.9412`,
  `effective_sharers ≈ 1.0625`.
- `bind_ns ≈ 93.0 + 15*0.0625 ≈ 93.94`.
- `interleave_ns = 90/4 + 190*3/4 = 165.0`.
- `93.94 < 165.0` → label is `"bind"`.

For an even split `{2500, 2500, 2500, 2500}`, `L = 0.25` exactly equals
the per-node average, so `base_bind_ns` alone would tie `interleave_ns`
at `165.0` -- but `effective_sharers = 4` (all four nodes share equally),
adding `15*3 = 45` ns of contention, so `bind_ns = 210.0 > 165.0` and the
label flips to `"interleave"`.

## What the gate checks

`main.cpp` runs a fixed battery of per-node access-count workloads
(across 4-node and 8-node configurations, from single-owner-dominated to
perfectly even) through `classify_workload` and prints `bind_ns`,
`interleave_ns`, and the winning label for each, to 3 decimal places.
The candidate's full stdout is compared byte-for-byte (`exact_match =
1.0`) against the reference's. Getting the contention term wrong (or
omitting it, which collapses every case to `"bind"` since pure latency
never favors interleave) changes both the printed numbers and, for the
evenly-shared workloads, the printed label.
